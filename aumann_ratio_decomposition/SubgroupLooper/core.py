import pysubgroup as ps
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import textwrap
from dataclasses import dataclass, asdict
from typing import List, Tuple

class AbsStdQFNumeric(ps.BoundedInterestingnessMeasure):
    """StandardQFNumeric のスコアを |…| にするだけの薄いラッパー"""
    def __init__(self, a=1.0, invert=False, estimator='sum'):
        self.base = ps.StandardQFNumeric(a=a, invert=invert, estimator=estimator)

    # ---- 以下は全部 base に委譲 ----
    def calculate_constant_statistics(self, data, target):
        self.base.calculate_constant_statistics(data, target)

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        return self.base.calculate_statistics(subgroup, target, data, statistics)

    def evaluate(self, subgroup, target, data, statistics=None):
        # ここだけ絶対値を取る
        return abs(self.base.evaluate(subgroup, target, data, statistics))

    def optimistic_estimate(self, subgroup, target, data, statistics=None):
        # 念のため絶対値（base 側は多くの場合正なのでそのままでもOK）
        return abs(self.base.optimistic_estimate(subgroup, target, data, statistics))


@dataclass
class SDConfig:
    """Configuration for subgroup discovery."""
    target_col: str = "aumann_shapley"
    use_abs_std_qf_numeric: bool = True
    depth: int = 5
    result_set_size: int = 1
    alpha_start: float = 0.1
    alpha_decay: float = 0.3
    alpha_end: float = 1
    min_support: int = 1
    ratio_threshold: float = 0.1
    beam_width: int = 20
    max_loops: int | None = None
    intervals_only: bool = False
    nbins: int = 5


class SubgroupLooper:
    """Iteratively extracts high‐impact subgroups and plots the summary.

    Parameters
    ----------
    data : pd.DataFrame
        Input data. *Must* contain ``cfg.target_col``.
    cfg : SDConfig
        Configuration.
    """

    def __init__(self, data: pd.DataFrame, cfg: SDConfig):
        self.data_orig = data.copy()
        self.cfg = cfg
        self.results: List[Tuple[pd.DataFrame, ps.Conjunction]] = []
        self.summary_df: pd.DataFrame | None = None

    # ---------------------------------------------------------------------
    # Core loop
    # ---------------------------------------------------------------------
    def run(self) -> pd.DataFrame:
        ret_prepped = self.data_orig.copy()
        target = ps.NumericTarget(self.cfg.target_col)
        alpha = self.cfg.alpha_start
        total_target = ret_prepped[self.cfg.target_col].abs().sum()

        loop = 0
        while len(ret_prepped) > 0:
            if self.cfg.max_loops is not None and loop >= self.cfg.max_loops:
                break

            print(f"[Loop {loop}] remaining={len(ret_prepped)}, alpha={alpha:.3f}")

            search_df = ret_prepped.drop(columns=[self.cfg.target_col])
            search_space = ps.create_selectors(search_df, nbins=self.cfg.nbins, intervals_only=self.cfg.intervals_only)

            qf = AbsStdQFNumeric(alpha) if self.cfg.use_abs_std_qf_numeric else ps.StandardQFNumeric(alpha)

            task = ps.SubgroupDiscoveryTask(
                data=ret_prepped,
                target=target,
                search_space=search_space,
                result_set_size=self.cfg.result_set_size,
                depth=self.cfg.depth,
                qf=qf,
                constraints=[ps.MinSupportConstraint(self.cfg.min_support)]
            )
            result = ps.BeamSearch(beam_width=self.cfg.beam_width).execute(task)

            entry = result.results[0]  # (quality, subgroup, stats)
            sg = entry[1]

            mask = sg.covers(ret_prepped)
            inside = ret_prepped[mask]
            outside = ret_prepped[~mask]

            self.results.append((inside, sg))

            contrib = inside[self.cfg.target_col].abs().sum()
            picked_ratio = contrib / total_target if total_target else 0
            remain_ratio = outside[self.cfg.target_col].abs().sum() / total_target if total_target else 0

            print(
                f"  -> picked {len(inside)} rows "
                f"({picked_ratio:.2%} of total target, {remain_ratio:.2%} remain), "
                f"condition: {sg}"
            )

            # Stop if residual is small enough
            if remain_ratio < self.cfg.ratio_threshold:
                print(
                    f"  -> remain ratio {remain_ratio:.2%} below threshold ({self.cfg.ratio_threshold:.2%}). stop."
                )
                if len(outside):
                    self.results.append((outside, ps.Conjunction([])))
                break

            ret_prepped = outside
            alpha += (self.cfg.alpha_end - alpha) * self.cfg.alpha_decay
            loop += 1

        self._summarize()
        return self.summary_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _summarize(self) -> None:
        rows: list[dict] = []
        for i, (df_sub, cond) in enumerate(self.results, 1):
            s = df_sub[self.cfg.target_col].sum()
            n = len(df_sub)
            rows.append({"rank": i, "rule": str(cond), "support": n, "sum": s})
        self.summary_df = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Plotting utils
    # ------------------------------------------------------------------
    def plot(
        self,
        df_vis: pd.DataFrame | None = None,
        *,
        bar_height: float = 0.6,
        wrap_labels: bool = True,
        wrap_width: int = 60,
        fig_width: float = 18.0,
        left_margin: float = 0.25,
        font_size: int = 8,
    ) -> None:
        """Visualise subgroup summary.

        Parameters
        ----------
        df_vis : pd.DataFrame, optional
            DataFrame returned by :py:meth:`run`.  If *None*, ``self.summary_df`` is used.
        bar_height : float, default 0.6
            Thickness of each horizontal bar.
        wrap_labels : bool, default True
            Whether to auto-wrap *rule* strings so long conditions do not overflow.
        wrap_width : int, default 30
            Number of characters per line when ``wrap_labels`` is *True*.
        fig_width : float, default 18.0
            Total width (in inches) of the whole figure.
        left_margin : float, default 0.25
            Fraction (0‒1) of the figure width to reserve for Y-axis labels.
            Increase this value if labels are clipped; decrease if you need wider graph area.
        """

        if df_vis is None:
            df_vis = self.summary_df
        if df_vis is None or df_vis.empty:
            print("Nothing to plot.")
            return

        df_plot = df_vis.sort_values("rank")
        y = np.arange(len(df_plot))
        sums = df_plot["sum"].values
        supports = df_plot["support"].values

        min_sum, max_sum = sums.min(), sums.max()
        pad = (max_sum - min_sum) * 0.1 if max_sum != min_sum else 0.1

        fig_height = max(8, len(df_plot) * (bar_height + 0.3))
        fig, ax1 = plt.subplots(figsize=(fig_width, fig_height))
        fig.subplots_adjust(left=left_margin)  # reserve space for labels

        colors = np.where(sums >= 0, "C0", "C3")
        ax1.barh(y, sums, color=colors, alpha=0.9, height=bar_height, zorder=2)

        # 0-line & helper grids
        ax1.axvline(0, color="k", linewidth=1)
        loc = ticker.MaxNLocator(nbins=6)
        ticks = loc.tick_values(min_sum, max_sum)
        for v in ticks:
            if abs(v) > 1e-12:
                ax1.axvline(v, color="gray", linestyle="--", linewidth=0.5, alpha=0.3, zorder=1)

        ax1.set_xlim(min_sum - pad, max_sum + pad)
        ax1.set_xlabel(f"Sum of {self.cfg.target_col}", fontsize=13, fontweight="bold")
        ax1.set_yticks(y)

        # optional label wrapping
        if wrap_labels:
            labels = [textwrap.fill(r, width=wrap_width) for r in df_plot["rule"]]
        else:
            labels = df_plot["rule"].tolist()
        ax1.set_yticklabels(labels, fontsize=font_size)
        ax1.invert_yaxis()

        # value annotations on bars
        for i, v in enumerate(sums):
            if v >= 0:
                ax1.text(v + pad * 0.015, i, f"{v:.3f}", va="center", ha="left", fontsize=9, color="C0")
            else:
                ax1.text(v - pad * 0.015, i, f"{v:.3f}", va="center", ha="right", fontsize=9, color="C3")

        # support line (top axis)
        ax2 = ax1.twiny()
        ax2.plot(supports, y, color="C1", marker="o", linewidth=2, zorder=3)
        ax2.set_xlim(0, supports.max() * 1.1 if supports.max() > 0 else 1)
        ax2.set_frame_on(False)
        ax2.tick_params(top=False, labeltop=False)
        ax1.text(
            1.02,
            1.02,
            "Support (#records)",
            transform=ax1.transAxes,
            color="C1",
            fontsize=10,
            ha="left",
            va="bottom",
        )

        # numeric support labels
        for i, v in enumerate(supports):
            ax2.text(v, i, f"{v:,}", va="center", ha="right", fontsize=8, color="C1")

        plt.title(f"Subgroups: {self.cfg.target_col} & support", fontsize=16, pad=18)
        plt.tight_layout()
        plt.show()