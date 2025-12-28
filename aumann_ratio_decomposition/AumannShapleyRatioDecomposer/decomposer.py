import itertools
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from collections import OrderedDict
from pandas.api.types import is_numeric_dtype

class AumannShapleyRatioDecomposer:
    """Compute Aumann–Shapley contributions for before/after ratio changes.

    The class supports two granularities:

    * **detail** – each original record becomes a contribution record.
      The contribution *dx/dy* is simply *\pm* the original value.
    * **group** – records are aggregated by ``keys`` in *before* and *after*,
      outer-joined, and the contribution is the aggregated difference
      (*new − prev*).

    In addition, the class supports two **ratio modes** that determine how
    the ratio change is conceptualized:

    * **"level"** (default) – raw replacement of numerator/denominator levels.
      Contributions are computed from level differences as-is.
    * **"composition"** – composition-share replacement. After attaching global
      totals (:math:`D_\ell, N_\ell, D_r, N_r`) per ``keys`` group, the *after*
      values are rescaled so that :math:`D_r` matches :math:`D_\ell`:
      for each record, :math:`(\text{den}_\text{aft}, \text{num}_\text{aft}) \gets
      ( \text{den}_\text{aft}, \text{num}_\text{aft}) \times (D_\ell / D_r)`,
      and likewise :math:`(D_r, N_r) \gets (D_r, N_r) \times (D_\ell / D_r)`.
      Then ``dx`` and ``dy`` are recomputed from these rescaled after-values.
      This isolates the effect of **share (composition) changes** from level changes.

    Parameters
    ----------
    df_before, df_after
        DataFrames that contain *denominator* and *numerator* columns.
    den_col, num_col
        Column names of the denominator and numerator.
    keys
        Columns that define the *global* aggregation level – i.e. how to
        compute :math:`D_\ell, N_\ell, D_r, N_r` that appear in the
        Aumann–Shapley formula. ``None`` means "treat the entire dataset as
        one group".
    mode
        ``"detail"`` or ``"group"``.
    ratio_mode
        ``"level"`` or ``"composition"`` (semantics of ratio change).
        In ``"composition"`` mode, per-``keys`` groups are normalized by the
        factor :math:`D_\ell/D_r` before contributions are computed.
    eps
        Positive constant added to denominators to avoid zero-division.
        If *None*, an automatic value (1 % of the smallest non-zero before/after
        denominator, clipped at ``1e-12``) is chosen.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self,
                 df_before: pd.DataFrame,
                 df_after: pd.DataFrame,
                 den_col: str,
                 num_col: str,
                 keys=None,
                 *,
                 mode="group",
                 ratio_mode="level",
                 eps: float = None):

        if df_before is None:
            if df_after is None:
                raise ValueError("Either df_before or df_after must be provided.")
            df_before = df_after.head(0).copy()

        # --- basic validation -------------------------------------------------
        for name, df in {"df_before": df_before, "df_after": df_after}.items():
            missing = [c for c in (den_col, num_col) if c not in df.columns]
            if missing:
                raise ValueError(f"{name} is missing required columns: {missing}")

        self.keys = ([keys] if isinstance(keys, str) else keys) or []
        self.mode = mode.lower()
        if self.mode not in {"detail", "group"}:
            raise ValueError("mode must be 'detail' or 'group'")

        self.ratio_mode = ratio_mode.lower()
        if self.ratio_mode not in {"level", "composition"}:
            raise ValueError("ratio_mode must be 'level' or 'composition'")

        # ---------------------------------------------------------------
        # ensure numerator / denominator are numeric
        #    – cast to float if necessary and log the action
        # ---------------------------------------------------------------
        df_before = df_before.copy()
        df_after  = df_after.copy()

        for df_name, df in [("df_before", df_before), ("df_after", df_after)]:
            for col in (den_col, num_col):
                if not is_numeric_dtype(df[col]):
                    print(f"[{df_name}] casting column '{col}' to float")
                    df[col] = df[col].astype(float)

        # ------------------------------------------------------------------
        # epsilon – 1 % of the minimum positive denominator, min 1e‑12
        # ------------------------------------------------------------------
        if eps is None:
            # Take the smallest non-zero value from the before/after denominator
            den_vals = pd.concat([df_before[den_col], df_after[den_col]]).to_numpy()
            nonzero = den_vals[den_vals > 0]
            eps = float(nonzero.min() * 0.01) if len(nonzero) > 0 else 1e-12
            if eps <= 1e-12:
                eps = 1e-12
        self.eps = eps

        self.den_col, self.num_col = den_col, num_col
        self.df_before_orig = df_before.copy()
        self.df_after_orig = df_after.copy()

        # ------------------------------------------------------------------
        # build records
        # ------------------------------------------------------------------
        if self.mode == "detail":
            self.records = self._make_detail(df_before, df_after)
        elif self.mode == "group":
            self.records = self._make_group(df_before, df_after)

        # attach global totals (Dl, Nl, Dr, Nr) to each record
        self._attach_totals(df_before, df_after)

        if self.ratio_mode == "composition":
            self._apply_composition_normalization()

        self.computed = False  # flag – whether contributions are computed

    def _apply_composition_normalization(self):
        """
        Re-scale after-values so that total denominator Dr matches Dl.
        This transforms level differences into composition-share differences.

        Steps:
        - Compute scaling factor = Dl / Dr (per group of keys).
        - Multiply after denominators, numerators, and global totals (Dr, Nr)
          by this factor.
        - Recompute dx, dy from the rescaled after-values.
        """

        den_bef_col = f"{self.den_col}_bef"
        num_bef_col = f"{self.num_col}_bef"
        den_aft_col = f"{self.den_col}_aft"
        num_aft_col = f"{self.num_col}_aft"

        # Scaling factor = Dl / Dr (avoid division by zero)
        Dl = self.records["D_l"].to_numpy(dtype=np.float64, copy=False)
        Dr = self.records["D_r"].to_numpy(dtype=np.float64, copy=False)
        scale = np.where(np.abs(Dr) > 0, Dl / (Dr + 0.0), 1.0)

        # Scale after-values (per record)
        self.records[den_aft_col] = self.records[den_aft_col].to_numpy(dtype=np.float64, copy=False) * scale
        self.records[num_aft_col] = self.records[num_aft_col].to_numpy(dtype=np.float64, copy=False) * scale

         # Scale global totals (Dr, Nr) consistently
        self.records["D_r"] = Dr * scale
        self.records["N_r"] = self.records["N_r"].to_numpy(dtype=np.float64, copy=False) * scale

        # Recalculate dx, dy after normalization
        self.records["dx"] = self.records[den_aft_col] - self.records[den_bef_col]
        self.records["dy"] = self.records[num_aft_col] - self.records[num_bef_col]

    # ------------------------------------------------------------------
    # detail mode – one record per original row (dx/dy = ± original value)
    # ------------------------------------------------------------------
    def _make_detail(self, bef, aft):
        """Return a DataFrame of contribution records for *detail* mode."""
        bef = bef.copy()
        aft = aft.copy()

        # Add line number + before/after as an aggregate key to process as detailed
        bef["_row"] = bef.index
        aft["_row"] = aft.index
        bef["_phase"] = "before"
        aft["_phase"] = "after"

        common_cols = list(set(bef.columns) & set(aft.columns))
        grp_cols = [c for c in common_cols if c not in (self.den_col, self.num_col)]
        

        # Aggregate before/after with keys. Nothing is actually done but run for consistency with group mode
        gb_bef = (bef.groupby(grp_cols, dropna=False, observed=True)
                    .agg({self.den_col: "sum", self.num_col: "sum"})
                    .reset_index())
        gb_aft = (aft.groupby(grp_cols, dropna=False, observed=True)
                    .agg({self.den_col: "sum", self.num_col: "sum"})
                    .reset_index())

        # Calculate the difference between the numerator and denominator
        merged = gb_bef.merge(gb_aft, on=grp_cols, how="outer",
                            suffixes=("_bef", "_aft")).fillna(0)
        merged["dx"] = merged[f"{self.den_col}_aft"] - merged[f"{self.den_col}_bef"]
        merged["dy"] = merged[f"{self.num_col}_aft"] - merged[f"{self.num_col}_bef"]

        # store before values so that each record has a baseline numerator/denominator
        merged[self.den_col] = merged[f"{self.den_col}_bef"]
        merged[self.num_col] = merged[f"{self.num_col}_bef"]

        return merged

    # ------------------------------------------------------------------
    # group mode – aggregate by keys, outer‑join, and diff
    # ------------------------------------------------------------------
    def _make_group(self, bef, aft):
        """Return a DataFrame of contribution records for *group* mode."""
        common_cols = list(set(bef.columns) & set(aft.columns))
        grp_cols = [c for c in common_cols if c not in (self.den_col, self.num_col)]

        if not grp_cols:
            # Add dummy column for groupby with one whole line
            bef = bef.copy()
            aft = aft.copy()
            bef["_dummy"] = 0
            aft["_dummy"] = 0
            grp_cols = ["_dummy"]

        # Aggregate numerator and denominator
        gb_bef = (bef.groupby(grp_cols, dropna=False, observed=True)
                        .agg({self.den_col: "sum", self.num_col: "sum"})
                        .reset_index())
        gb_aft = (aft.groupby(grp_cols, dropna=False, observed=True)
                        .agg({self.den_col: "sum", self.num_col: "sum"})
                        .reset_index())

        # Calculate the difference between the numerator and denominator
        merged = gb_bef.merge(gb_aft, on=grp_cols, how="outer",
                            suffixes=("_bef", "_aft"))
        
        # Fill only non-categorical columns (old pandas bug: .fillna on Categorical with new value could cause RecursionError)
        cat_cols   = merged.select_dtypes(include="category").columns        # Categorical only
        other_cols = merged.columns.difference(cat_cols)                     # everything else
        merged[other_cols] = merged[other_cols].fillna(0)

        merged["dx"] = merged[f"{self.den_col}_aft"] - merged[f"{self.den_col}_bef"]
        merged["dy"] = merged[f"{self.num_col}_aft"] - merged[f"{self.num_col}_bef"]

        # store before values so that each record has a baseline numerator/denominator
        merged[self.den_col] = merged[f"{self.den_col}_bef"]
        merged[self.num_col] = merged[f"{self.num_col}_bef"]

        return merged

    # ------------------------------------------------------------------
    # attach global totals (Dl, Nl, Dr, Nr) per *keys* group
    # ------------------------------------------------------------------
    def _attach_totals(self, bef, aft):
        if self.keys:
            tot_bef = (bef.groupby(self.keys, dropna=False, observed=True)
                           .agg(D_l=(self.den_col, "sum"),
                                N_l=(self.num_col, "sum"))
                           .reset_index()).fillna(0)
            tot_aft = (aft.groupby(self.keys, dropna=False, observed=True)
                           .agg(D_r=(self.den_col, "sum"),
                                N_r=(self.num_col, "sum"))
                           .reset_index())
            totals = tot_bef.merge(tot_aft, on=self.keys, how="outer").fillna(0)
        else:
            totals = pd.DataFrame(dict(
                D_l=[bef[self.den_col].sum()], N_l=[bef[self.num_col].sum()],
                D_r=[aft[self.den_col].sum()], N_r=[aft[self.num_col].sum()]))
            totals["_dummy"] = 0
            self.records["_dummy"] = 0
            self.keys = ["_dummy"]

        self.records = self.records.merge(totals, on=self.keys, how="left")

    # ------------------------------------------------------------------
    # 1. Aumann–Shapley value
    # ------------------------------------------------------------------
    def compute_aumann_shapley(self):
        """Compute and return the aumann_shapley contribution for each record."""
        rec = self.records

        # Extract columns as NumPy arrays (avoid Python-level loops)
        Dl = rec["D_l"].to_numpy(dtype=np.float64, copy=False)
        Dr = rec["D_r"].to_numpy(dtype=np.float64, copy=False)
        Nl = rec["N_l"].to_numpy(dtype=np.float64, copy=False)
        Nr = rec["N_r"].to_numpy(dtype=np.float64, copy=False)
        Di = rec["dx"].to_numpy(dtype=np.float64, copy=False)
        Ni = rec["dy"].to_numpy(dtype=np.float64, copy=False)

        eps = float(self.eps)
        mask = np.minimum(Dl, Dr) < eps
        Dl = Dl + eps * mask
        Dr = Dr + eps * mask

        dD = Dr - Dl
        dN = Nr - Nl

        out = np.empty_like(Dl, dtype=np.float64)
        
        # Mask for records where |ΔD| is very small (special-case handling)
        mask = np.abs(dD) < eps
        # case1: ΔD ≈ 0
        out[mask] = (Ni[mask] / Dl[mask]) - (Di[mask] * (Nl[mask] + Nr[mask])) / (2.0 * Dl[mask]**2)

        # Case 2: General formula
        m = ~mask
        s = dN[m] / dD[m]
        C = Nl[m] - s * Dl[m]
        # More stable computation: log(Dr/Dl) = log(Dr) - log(Dl)
        log_term = np.log(Dr[m]) - np.log(Dl[m])
        inv_diff = (1.0 / Dl[m]) - (1.0 / Dr[m])

        out[m] = (Ni[m] * log_term - Di[m] * (C * inv_diff + s * log_term)) / dD[m]

        # Store result back to DataFrame
        rec["aumann_shapley"] = pd.Series(out, index=rec.index)
        self.computed = True
        return rec["aumann_shapley"]

    # ------------------------------------------------------------------
    # 2. Exact Shapley value – factorial complexity, use only for small N!
    # ------------------------------------------------------------------
    def compute_exact(self):
        """Compute the *exact* Shapley value. **O(n!**) – use for small groups."""
        if self.mode == "detail":
            raise ValueError(
                "compute_exact is not supported in 'detail' mode. "
                "Use 'group' mode instead."
            )
        
        exact_all = np.zeros(len(self.records))

        if self.keys:
            for _, idx in self.records.groupby(self.keys, sort=False).groups.items():
                idx = list(idx) 
                nums0 = self.records.loc[idx, f"{self.num_col}_bef"].to_numpy()
                dens0 = self.records.loc[idx, f"{self.den_col}_bef"].to_numpy()
                    
                dx_arr = self.records.loc[idx, "dx"].to_numpy()
                dy_arr = self.records.loc[idx, "dy"].to_numpy()

                exact_all[idx] = self._exact_shapley(nums0, dens0, dx_arr, dy_arr)
        else:   # Treat the entire group as a single group
            nums0 = self.records[f"{self.num_col}_bef"].to_numpy()
            dens0 = self.records["{self.den_col}_bef"].to_numpy()
            dx_arr = self.records["dx"].to_numpy()
            dy_arr = self.records["dy"].to_numpy()

            exact_all[:] = self._exact_shapley(nums0, dens0, dx_arr, dy_arr)

        self.records["shapley_exact"] = exact_all
        return exact_all

    @staticmethod
    def _exact_shapley(nums0, dens0, dx_arr, dy_arr):
        """Compute the exact Shapley value by enumerating all permutations.

        Parameters
        ----------
        nums_before : np.ndarray
            Initial numerators of all records in the group.
        dens_before : np.ndarray
            Initial denominators of all records in the group.
        dx_arr : np.ndarray
            Changes in denominators for each record.
        dy_arr : np.ndarray
            Changes in numerators for each record.

        Returns
        -------
        np.ndarray
            Exact Shapley values for each record.
            Complexity is O(n!), so only feasible for small n.
        """
        n = len(nums0)
        exact = np.zeros(n)
        perms = itertools.permutations(range(n))
        for perm in perms:
            nums = nums0.copy()
            dens = dens0.copy()
            if dens.sum() == 0:
                r_cur = 0
            else:
                r_cur = nums.sum()/dens.sum()
            
            for idx in perm:
                nums[idx] += dy_arr[idx]
                dens[idx] += dx_arr[idx]
                if dens.sum() == 0:
                    r_new = 0
                else:
                    r_new = nums.sum()/dens.sum()
                exact[idx] += r_new - r_cur
                r_cur = r_new
        exact /= math.factorial(n)
        return exact

    # -----------------------------
    # 4) Compute and return results
    # -----------------------------
    def result(self, split: bool = True, drop_internal=True):
        """Return computed contributions as DataFrame(s).

        In *group* mode:
            Returns a single DataFrame of aggregated records.

        In *detail* mode:
            - If ``split=True``: returns a tuple of (before_df, after_df),
              each merged with the computed Shapley values.
            - If ``split=False``: returns a single combined DataFrame.

        Parameters
        ----------
        split : bool, default True
            Whether to split *detail* mode output into (before, after).
            Ignored in *group* mode.
        drop_internal : bool, default True
            Whether to drop internal columns used for computation
            (e.g., dx/dy, totals).

        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
            DataFrame(s) containing the results with Shapley contributions.
        """
        if not self.computed:
            self.compute_aumann_shapley()

        if self.mode == "detail" and split:
            bef = self.records[self.records["_phase"] == "before"]
            aft = self.records[self.records["_phase"] == "after"]

            # Merge Shapley values back into the original before/after DataFrames
            bef_out = self.df_before_orig.merge(
                bef[["_row", "aumann_shapley"]],
                left_index=True, right_on="_row", how="left").drop(columns="_row")
            aft_out = self.df_after_orig.merge(
                aft[["_row", "aumann_shapley"]],
                left_index=True, right_on="_row", how="left").drop(columns="_row")

            if drop_internal:
                return bef_out, aft_out
            else:
                return bef, aft

        # Group mode or unsplit detail mode
        if drop_internal:
            if self.mode == "detail":
                return self.records.drop(columns=["_phase", "_row", "dx", "dy", "D_l", "N_l", "D_r", "N_r"])
            else:
                return self.records.drop(columns=["dx", "dy", "D_l", "N_l", "D_r", "N_r"])
        else:
            return self.records.copy()

    def analyze(self):
        """Run a quick diagnostic analysis with default visualizations.

        This method automatically computes Shapley contributions if not
        already computed, then generates:
        - Shapley value rank plot
        - Shapley value histogram
        - Correlation heatmap with numeric columns
        - Ratio vs. Shapley value comparison by keys
        """
        if self.computed is False:
            self.compute_aumann_shapley()

        self._plot_shapley_rank()
        self._plot_shapley_histogram()
        self._plot_correlations()
        self._plot_ratio_vs_shapley_by_keys()

    def _plot_shapley_rank(
            self,
            *,
            shapley_col: str = "aumann_shapley",
            title = None,
            figsize: tuple = (6, 4),
            color = None):
        """Plot Shapley values sorted by level value (rank vs. value).

        Parameters
        ----------
        shapley_col : str, default "aumann_shapley"
            Column containing Shapley values to plot 
            (e.g., "aumann_shapley" or "shapley_exact").
        title : str or None, default None
            Plot title. If None, a default title is generated.
        figsize : tuple, default (6, 4)
            Size of the matplotlib figure.
        color : str or None, default None
            Line color for the plot.
        """
        if self.computed is False:
            self.compute_aumann_shapley()

        if shapley_col not in self.records.columns:
            raise ValueError(
                f"Column '{shapley_col}' not found. "
            )

        # Extract and sort values
        vals = self.records[shapley_col].dropna().to_numpy()
        idx_sorted = np.argsort(vals)  # 値の昇順に並べる
        vals_sorted = vals[idx_sorted]

        # plot
        plt.figure(figsize=figsize)
        plt.plot(range(1, len(vals_sorted) + 1), vals_sorted,
                marker="", linestyle="-", color=color)
        plt.xlabel("Rank (abs. descending)")
        plt.ylabel(shapley_col)
        plt.title(title or f"Rank plot of {shapley_col}")
        plt.grid(alpha=0.3)
        plt.show()

    def _plot_shapley_histogram(
        self,
        *,
        shapley_col: str = "aumann_shapley",
        bins: int = 100,
        figsize: tuple = (10, 4),
        color = None,
        title = None,
        density: bool = False,
        alpha: float = 0.7,
        zoom_iqr_multiplier: float = 2.0
    ):
        """Plot histograms of Shapley values (full and zoomed by IQR).

        This function creates two side-by-side histograms:
        - Left: Full value range
        - Right: Zoomed range based on IQR around the median

        Parameters
        ----------
        shapley_col : str, default "aumann_shapley"
            Column containing Shapley values to plot.
        bins : int, default 100
            Number of histogram bins.
        figsize : tuple, default (10, 4)
            Size of the matplotlib figure.
        color : str or None, default None
            Bar color for the histogram.
        title : str or None, default None
            Title for the full histogram plot.
        density : bool, default False
            If True, normalize histogram to show probability density.
        alpha : float, default 0.7
            Transparency of the histogram bars.
        zoom_iqr_multiplier : float, default 2.0
            Multiplier for IQR range used in zoomed histogram.
        """
        if self.computed is False:
            self.compute_aumann_shapley()

        if shapley_col not in self.records.columns:
            raise ValueError(
                f"Column '{shapley_col}' not found. "
            )

        vals = self.records[shapley_col].dropna().to_numpy()

        # Determine zoomed range using IQR
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = q75 - q25
        q_low = q25 - zoom_iqr_multiplier * iqr
        q_high = q75 + zoom_iqr_multiplier * iqr

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        # Full histogram
        axes[0].hist(vals,
                    bins=bins,
                    density=density,
                    alpha=alpha,
                    color=color,
                    edgecolor="black")
        axes[0].set_title(title or f"Histogram of {shapley_col} (Full)")
        axes[0].set_xlabel(shapley_col)
        axes[0].set_ylabel("Density" if density else "Frequency")
        axes[0].grid(alpha=0.3)

        # Zoomed histogram (IQR-based)
        axes[1].hist(vals,
                    bins=bins,
                    range=(q_low, q_high),
                    density=density,
                    alpha=alpha,
                    color=color,
                    edgecolor="black")
        axes[1].set_title(f"Zoomed (median±{zoom_iqr_multiplier}×IQR)")
        axes[1].set_xlabel(shapley_col)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    def _plot_correlations(
        self,
        *,
        shapley_col: str = "aumann_shapley",
        numeric_cols: list[str] | None = None,
        dropna: bool = True
    ) -> pd.Series:
        """Plot and compute correlations between Shapley values and numeric columns.

        This method computes Pearson correlation coefficients between the
        specified numeric columns and the selected Shapley column, then
        displays them as a heatmap.

        Parameters
        ----------
        shapley_col : str, default "aumann_shapley"
            Column containing Shapley values to analyze.
        numeric_cols : list[str] or None, default None
            List of numeric columns to correlate with Shapley values.
            If None, all numeric columns (excluding shapley_col) are used.
        dropna : bool, default True
            Whether to drop rows with NaN in the shapley_col.

        Returns
        -------
        pd.Series
            Pearson correlation coefficients for each numeric column.
        """
        if self.computed is False:
            self.compute_aumann_shapley()

        df = self.records.copy()
        if shapley_col not in df.columns:
            raise ValueError(
                f"Column '{shapley_col}' not found. "
            )
        
        # Drop NaNs if requesteds
        if dropna:
            df = df.dropna(subset=[shapley_col])

        # Determine numeric columns if not specified
        if numeric_cols is None:
            numeric_cols = [
                c for c in df.select_dtypes(include=[np.number]).columns
                if c != shapley_col
            ]

        # Compute correlations and plot heatmap
        corr = pd.Series(dtype=float)
        if numeric_cols:
            corr = df[numeric_cols].corrwith(df[shapley_col])

            plt.figure(figsize=(max(6, len(numeric_cols)*0.4), 2))
            sns.heatmap(
                corr.to_frame().T,
                annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=False,
                linewidths=0.5, linecolor="gray"
            )
            plt.title(f"Correlation with {shapley_col}")
            plt.yticks([])
            plt.tight_layout()
            plt.show()

        return corr

    def _plot_ratio_vs_shapley_by_keys(
        self,
        shapley_col: str = "aumann_shapley",
        ratio_digits: int = 6,
        show_table: bool = False,
        figsize: tuple = (10, 5)
    ) -> pd.DataFrame:
        """Compare ratio changes vs. Shapley contributions grouped by keys.

        This function:
        - Aggregates before/after numerator-denominator ratios by keys
        - Computes their difference (after - before)
        - Sums Shapley contributions by keys
        - Computes residual error (ratio_diff - shapley_sum)
        - Sorts by ratio difference
        - Plots a dual-axis line chart comparing these metrics

        Parameters
        ----------
        shapley_col : str, default "aumann_shapley"
            Column containing Shapley values to plot.
        ratio_digits : int, default 6
            Number of decimal places to round ratio values.
        show_table : bool, default False
            Whether to display the resulting aggregated table.
        figsize : tuple, default (10, 5)
            Size of the matplotlib figure.

        Returns
        -------
        pd.DataFrame
            DataFrame with keys, ratio differences, Shapley sums, and error terms.
        """
        if self.computed is False:
            self.compute_aumann_shapley()

        # Prepare before/after values
        df = self.records.copy()
        all_df = df.copy()
        for col in [self.den_col, self.num_col]:
            if f"{col}_bef" not in all_df:
                all_df[f"{col}_bef"] = all_df[col]
            if f"{col}_aft" not in all_df:
                all_df[f"{col}_aft"] = all_df[col] + all_df.get("dx" if col==self.den_col else "dy", 0)

        # Aggregate by keys
        grp = all_df.groupby(self.keys, dropna=False, observed=True).agg({
            f"{self.den_col}_bef": "sum", f"{self.num_col}_bef": "sum",
            f"{self.den_col}_aft": "sum", f"{self.num_col}_aft": "sum",
            shapley_col: "sum"
        }).reset_index()

        # Compute ratios and differences
        grp["ratio_bef"] = grp[f"{self.num_col}_bef"] / grp[f"{self.den_col}_bef"]
        grp["ratio_aft"] = grp[f"{self.num_col}_aft"] / grp[f"{self.den_col}_aft"]
        grp["ratio_diff"] = grp["ratio_aft"] - grp["ratio_bef"]

        # Shapley sum and residual error
        grp["shapley_sum"] = grp[shapley_col]
        grp["error_term"] = grp["ratio_diff"] - grp["shapley_sum"]

        # Round numeric columns
        for c in ["ratio_bef", "ratio_aft", "ratio_diff", "shapley_sum", "error_term"]:
            grp[c] = grp[c].round(ratio_digits)

        # Sort by ratio difference
        grp = grp.sort_values("ratio_diff").reset_index(drop=True)

        # Plot line chart (dual axis)
        plt.figure(figsize=figsize)
        x = range(len(grp))
        plt.plot(x, grp["ratio_diff"], marker="o", label="ratio_diff (aft-bef)")
        plt.plot(x, grp["shapley_sum"], marker="o", label="shapley_sum")
        ax = plt.gca()
        ax2 = ax.twinx()
        ax2.plot(x, grp["error_term"], marker="x", linestyle="--", color="gray", label="error_term (2nd axis)")
        ax.set_xticks(x)
        ax.set_xticklabels(grp[self.keys[0]].astype(str) if len(self.keys) == 1 else grp[self.keys].astype(str).agg("_".join, axis=1), rotation=45)
        ax.set_ylabel("ratio_diff / shapley_sum")
        ax2.set_ylabel("error_term")
        ax.set_title(f"By keys: Ratio Difference, Shapley Sum, Error (Dual Axes)")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.show()

        if show_table:
            display(grp)

        return grp

    def plot_ratio_with_shapley_stacked(
        self,
        x_key: str,
        decompose_keys: list[str],
        *,
        filter_keys = None,
        shapley_col: str = "aumann_shapley",
        figsize: tuple = (12, 6),
        bar_alpha: float = 0.8,
        bar_width: float = 0.55,
        residual_color: str = "lightgray",
        vline: bool = False,
        vline_style = None,
        ratio_ylim = None,
        shap_ylim = None,
        legend_ncol: int = 6,
    ):
        """Plot ratio transitions and Shapley decomposition as a stacked bar chart.

        This function:
        - Plots before/after ratios as a line chart (primary y-axis)
        - Decomposes the Shapley contributions by `decompose_keys` as stacked bars (secondary y-axis)
        - Adds a residual bar to ensure that stacked contributions + residual = ratio_diff

        Parameters
        ----------
        x_key : str
            Column to use as the x-axis.
        decompose_keys : list
            Columns to use for Shapley decomposition (stacked bar categories).
        filter_keys : dict or None, optional
            Dictionary of filters {col: value or [values]} applied before aggregation.
            If None, all combinations of other keys (excluding `x_key`) are automatically used.
        shapley_col : str, default "aumann_shapley"
            Column containing Shapley contributions.
        figsize : tuple, default (12, 6)
            Figure size.
        bar_alpha : float, default 0.8
            Transparency for bars.
        bar_width : float, default 0.55
            Width of the bars.
        bar_colors : dict, optional
            Custom color mapping for stacked bars.
        residual_color : str, default "lightgray"
            Color for the residual bar.
        vline : bool, default False
            Whether to draw vertical separator lines between x-axis categories.
        vline_style : dict, optional
            Styling for the vertical separator lines.
        ratio_ylim : tuple or None, optional
            Tuple ``(ymin, ymax)`` to fix the y-range of the ratio lines (primary axis).
            If None (default), Matplotlib decides automatically.
        shap_ylim : tuple or None, optional
            Tuple ``(ymin, ymax)`` to fix the y-range of the stacked Shapley bars
            (secondary axis).  If None (default), auto-scale.
        legend_ncol : int, optional
            Number of columns to use in the stacked bar legend.
        Returns
        -------
        tuple
            (grp_ratio, shap_pivot, residual, fig, (ax1, ax2)):
            - grp_ratio: DataFrame with before/after ratios and differences.
            - shap_pivot: Pivoted DataFrame of Shapley contributions by decompose_keys.
            - residual: Series of residual errors (ratio_diff - sum of Shapley).
            - fig, (ax1, ax2): Figure and Axes objects.

        Examples
        --------
        >>> # Initialize the decomposer with before/after data
        >>> decomposer = AumannShapleyRatioDecomposer(
        ...     df_before, df_after,
        ...     keys=["month", "product", "region"],
        ...     den_col="policy_count", num_col="loss"
        ... )
        >>> decomposer.compute_aumann_shapley()

        >>> # (1) Plot ratio transition by month, decomposed by product, filtering region
        >>> decomposer.plot_ratio_with_shapley_stacked(
        ...     x_key="month",
        ...     decompose_keys=["product"],
        ...     filter_keys={"region": ["East", "West"]},
        ...     vline=True
        ... )

        >>> # (2) Plot without filter_keys: automatically enumerates all combinations
        >>> #     of remaining keys (here: product × region) and plots them.
        >>> #     Be careful: if there are many combinations, many charts will be generated.
        >>> decomposer.plot_ratio_with_shapley_stacked(
        ...     x_key="month",
        ...     decompose_keys=["product"]
        ... )
        """

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        if not self.computed:
            self.compute_aumann_shapley()

        # Auto-generate filter_keys if None
        if filter_keys is None:
            other_keys = [k for k in self.keys if k != x_key and k != "_dummy"]
            if other_keys:
                unique_values = [self.records[k].dropna().unique() for k in other_keys]
                filter_keys_list = [
                    dict(zip(other_keys, values)) for values in product(*unique_values)
                ]
            else:
                filter_keys_list = [None]
        else:
            filter_keys_list = [filter_keys]

        # Validate keys: x_key + filter_keys must match self.keys
        results = OrderedDict()
        for fks in filter_keys_list:
            used_keys = {x_key} | (set(fks.keys()) if fks else set())
            defined_keys = {k for k in self.keys if k != "_dummy"}
            if used_keys != defined_keys:
                missing = defined_keys - used_keys
                extra   = used_keys - defined_keys
                msg = []
                if missing: msg.append(f"Missing: {missing}")
                if extra:   msg.append(f"Extra: {extra}")
                raise ValueError(
                    "x_key and filter_keys must match self.keys.\n"
                    f"self.keys = {sorted(defined_keys)}\n"
                    f"Used keys = {sorted(used_keys)}\n" + "\n".join(msg)
                )

            df = self.records.copy()

            # Create decomposition key
            if not decompose_keys:
                decompose_key = "_dummy"
                df[decompose_key] = 0
            elif len(decompose_keys) == 1:
                decompose_key = decompose_keys[0]
            else:
                decompose_key = "|".join(decompose_keys)
                df[decompose_key] = df[decompose_keys].astype(str).agg("|".join, axis=1)

            # Rebuild before/after columns
            all_df = df.copy()
            for c in [self.den_col, self.num_col]:
                if f"{c}_bef" not in all_df:
                    all_df[f"{c}_bef"] = all_df[c]
                if f"{c}_aft" not in all_df:
                    all_df[f"{c}_aft"] = all_df[c] + all_df.get("dx" if c==self.den_col else "dy", 0)
            df_for_shap = df

            # Remove records where both before & after numerator/denominator are all zero
            def _filter_zero_records(d):
                return d[~(
                    (d[f"{self.den_col}_bef"] == 0) & (d[f"{self.num_col}_bef"] == 0) &
                    (d[f"{self.den_col}_aft"] == 0) & (d[f"{self.num_col}_aft"] == 0)
                )].copy()

            all_df      = _filter_zero_records(all_df)
            df_for_shap = _filter_zero_records(df_for_shap)

            # Apply custom filters
            def _apply_filter(d):
                if not fks:
                    return d
                for k, v in fks.items():
                    d = d[d[k].isin(v)] if isinstance(v, (list, tuple, set)) else d[d[k] == v]
                return d

            all_df      = _apply_filter(all_df)
            df_for_shap = _apply_filter(df_for_shap)

            def safe_ratio(num, den):
                return np.where(den != 0, num / den, 0)

            # Group ratios
            den_bef_col = f"{self.den_col}_bef"
            num_bef_col = f"{self.num_col}_bef"
            den_aft_col = f"{self.den_col}_aft"
            num_aft_col = f"{self.num_col}_aft"

            grp_ratio = (
                all_df.groupby(x_key, dropna=False, observed=True)
                .agg(**{
                    den_bef_col: (den_bef_col, 'sum'),
                    num_bef_col: (num_bef_col, 'sum'),
                    den_aft_col: (den_aft_col, 'sum'),
                    num_aft_col: (num_aft_col, 'sum')
                })
                .reset_index()
            )
            grp_ratio["ratio_bef"]  = safe_ratio(grp_ratio[f"{self.num_col}_bef"], grp_ratio[f"{self.den_col}_bef"])
            grp_ratio["ratio_aft"]  = safe_ratio(grp_ratio[f"{self.num_col}_aft"], grp_ratio[f"{self.den_col}_aft"])
            grp_ratio["ratio_diff"] = grp_ratio["ratio_aft"] - grp_ratio["ratio_bef"]

            # Aggregate Shapley contributions
            shap_grp = (df_for_shap.groupby([x_key, decompose_key], dropna=False, observed=True)[shapley_col]
                        .sum()
                        .reset_index())
            shap_pivot = shap_grp.pivot(index=x_key, columns=decompose_key,
                                        values=shapley_col).fillna(0)

            # Align indexes
            idx_all = pd.Index(grp_ratio[x_key]).union(shap_pivot.index)
            grp_ratio  = grp_ratio.set_index(x_key).reindex(idx_all).fillna(0).reset_index()
            shap_pivot = shap_pivot.reindex(idx_all).fillna(0)

            shap_sum  = shap_pivot.sum(axis=1)
            residual  = grp_ratio.set_index(x_key)["ratio_diff"] - shap_sum

            # ========== Visualization ==========
            x_vals = np.arange(len(idx_all))
            fig, ax1 = plt.subplots(figsize=figsize)

            # Plot ratio lines
            l1, = ax1.plot(x_vals, grp_ratio["ratio_bef"], marker='o', label='ratio_bef', zorder=4)
            l2, = ax1.plot(x_vals, grp_ratio["ratio_aft"], marker='o', label='ratio_aft', zorder=4)

            # Optional vertical separators
            if vline:
                if vline_style is None:
                    vline_style = {"color": "#aaaaaa", "linewidth": .6, "linestyle": "--", "alpha": .6}
                for xv in x_vals:
                    ax1.axvline(x=xv - 0.5, **vline_style)

            if decompose_keys:
                # Stacked bars (positive/negative separated)
                ax2 = ax1.twinx()
                ax2.axhline(0, color="#d62728", linewidth=1.2, linestyle="--", zorder=5)

                cols = shap_pivot.columns.tolist()
                cmap = plt.get_cmap("tab20")
                hatch_cycle = ['', '//', r'\\', '-', '+', 'x', '.', 'o', '*']
                bar_colors = {}
                bar_hatches = {}
                for i, col in enumerate(cols):
                    bar_colors[col] = cmap(i % cmap.N)
                    bar_hatches[col] = hatch_cycle[(i // cmap.N) % len(hatch_cycle)]

                pos_bottoms = np.zeros(len(idx_all))
                neg_bottoms = np.zeros(len(idx_all))
                bar_handles = []

                for col in shap_pivot.columns:
                    y = shap_pivot[col].to_numpy()
                    bottoms = np.where(y >= 0, pos_bottoms, neg_bottoms)
                    h = ax2.bar(
                        x_vals, y, bottom=bottoms, width=bar_width,
                        alpha=bar_alpha, color=bar_colors[col],
                        edgecolor="white", linewidth=0.4,
                        hatch=bar_hatches[col],
                        label=f"{decompose_key}: {col}" if y.any() else "_nolegend_",
                        zorder=2
                    )
                    bar_handles.append(h)

                    pos_bottoms += np.where(y > 0, y, 0)
                    neg_bottoms += np.where(y < 0, y, 0)

                # Residual bar
                res_bottoms = np.where(residual >= 0, pos_bottoms, neg_bottoms)
                h_res = ax2.bar(
                    x_vals, residual.to_numpy(), bottom=res_bottoms, width=bar_width,
                    color=residual_color, edgecolor="gray", linewidth=0.4,
                    alpha=0.9, label="Residual", zorder=2
                )
                ax2.set_ylabel("Shapley contribution", fontsize=11)

                if shap_ylim is not None:
                    ax2.set_ylim(*shap_ylim)

                ax2.grid(alpha=0.3, axis="y")
                ax1.set_zorder(ax2.get_zorder() + 1)

                handles2, labels2 = [], []
                for h in bar_handles:
                    handles2.append(h[0])
                    labels2.append(h.get_label())
                if len(h_res) > 0:
                    handles2.append(h_res[0])
                    labels2.append("Residual")

                fig.legend(
                    handles2, labels2,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.05),
                    ncol=legend_ncol,
                    fontsize=9
                )
            else:
                ax2 = None

            # Axes configuration
            ax1.patch.set_visible(False)
            ax1.set_xticks(x_vals)
            ax1.set_xticklabels(idx_all.astype(str), rotation=45, ha='right')
            ax1.set_xlabel(x_key, fontsize=12, fontweight="bold")
            ax1.set_ylabel("Ratio", fontsize=11)

            if ratio_ylim is not None:
                ax1.set_ylim(*ratio_ylim)


            # Title
            filter_str = ""
            if fks:
                filter_str = " | " + ", ".join(f"{k}={v}" for k, v in fks.items())
            if decompose_keys:
                title = f"Ratio transition & Shapley decomposition\nby {decompose_key}{filter_str}"
            else:
                title = f"Ratio transition by {filter_str}"
            ax1.set_title(title, fontsize=14, fontweight="bold")

            # Legends
            leg1 = ax1.legend(handles=[l1, l2], loc="upper left", bbox_to_anchor=(0, 1.05))
            ax1.add_artist(leg1)

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()

            results[tuple(sorted(fks.items())) if fks else None] = {
                "grp_ratio": grp_ratio,
                "shap_pivot": shap_pivot,
                "residual": residual,
                "fig": fig,
                "axes": (ax1, ax2),
            }

        return results
