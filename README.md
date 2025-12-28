# aumann_ratio_decomposition

Utilities to decompose ratio changes (numerator/denominator) using Aumann-Shapley
contributions, plus an iterative subgroup discovery helper.

## Features
- Aumann-Shapley ratio decomposition with group or detail granularity.
- Optional "composition" mode to isolate share effects.
- Plot helpers for ratio transitions and Shapley breakdowns.
- Subgroup discovery loop for highlighting high-impact rules.

## Requirements
- Python - 3.12 (Python 3.13+ is not supported).
- NumPy < 2.0 (required because `pysubgroup` does not support NumPy 2.x).

## Installation
```bash
pip install -e .
```

## Quickstart
```python
import pandas as pd
from aumann_ratio_decomposition import AumannShapleyRatioDecomposer

df_before = pd.DataFrame(
    {"den": [100, 200], "num": [10, 40], "group": ["A", "B"]}
)
df_after = pd.DataFrame(
    {"den": [120, 180], "num": [15, 38], "group": ["A", "B"]}
)

decomposer = AumannShapleyRatioDecomposer(
    df_before=df_before,
    df_after=df_after,
    den_col="den",
    num_col="num",
    keys=["group"],
    mode="group",          # or "detail"
    ratio_mode="level",    # or "composition"
)

result = decomposer.result()
print(result.head())
```

## Subgroup discovery
```python
from aumann_ratio_decomposition import SDConfig, SubgroupLooper

cfg = SDConfig(
    target_col="aumann_shapley",
    depth=3,
    min_support=1000,
    ratio_threshold=0.1,
    nbins=5,
)

runner = SubgroupLooper(result, cfg)
summary = runner.run()
runner.plot(summary, wrap_width=40, fig_width=12, font_size=16)
```

## Examples
- Notebooks are under `examples/` (e.g., `examples/example1.ipynb`).
