"""Regenerate the paper's comparison and age-contribution figures.

Run from the repository root:
  python images_for_paper/reproduce_revision_figures.py --csv csv/Kaggle.csv --output generated
Only the relevant notebook cells are executed. Notebook outputs are not rewritten.
"""
from pathlib import Path
import argparse
import json
import os
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def histogram(output):
    import matplotlib.pyplot as plt
    import numpy as np
    notebook = json.loads((ROOT / 'images_for_paper/create_image.ipynb').read_text(encoding='utf-8'))
    source = ''.join(notebook['cells'][1]['source'])
    figures = []
    show = plt.show
    plt.show = lambda: figures.append(plt.gcf())
    namespace = {}
    try:
        exec(compile(source, 'create_image.ipynb:cell1', 'exec'), namespace)
    finally:
        plt.show = show
    for i, fig in enumerate(figures):
        fig.savefig(output / ('random_shapey.png' if i == 0 else 'random_shapley_record2.png'), dpi=160, bbox_inches='tight')
    np.savez_compressed(output / 'histogram_values.npz', differences=np.asarray(namespace['diffs']))
    plt.close('all')


def age_figures(csv, output):
    import pandas as pd
    import matplotlib.pyplot as plt
    from aumann_ratio_decomposition import AumannShapleyRatioDecomposer
    notebook = json.loads((ROOT / 'examples/example1.ipynb').read_text(encoding='utf-8'))
    source = ''.join(notebook['cells'][2]['source'])
    source = source.replace('path = "../csv/Kaggle.csv"', 'path = input_csv')
    namespace = {'pd': pd, 're': re, 'os': os, 'input_csv': str(csv)}
    exec(compile(source, 'example1.ipynb:cell2', 'exec'), namespace)
    decomposer = AumannShapleyRatioDecomposer(namespace['df_f'], namespace['df_m'], mode='group',
                                           keys=['POLICY_TYPE_3', 'Policy_Year'], den_col='den', num_col='num')
    figures = decomposer.plot_ratio_with_shapley_stacked(x_key='Policy_Year',
                decompose_keys=['ENTRY_AGE_RANK'], figsize=(10, 6), legend_ncol=4)
    for key, result in figures.items():
        product = dict(key)['POLICY_TYPE_3']
        if product not in ('A', 'B'):
            continue
        name = 'shap_by_entry_age.png' if product == 'A' else 'shap_by_entry_age_B.png'
        result['fig'].savefig(output / name, dpi=160, bbox_inches='tight')
        result['grp_ratio'].to_csv(output / f'ratio_{product}.csv')
        result['shap_pivot'].to_csv(output / f'contributions_{product}.csv')
    plt.close('all')
    return decomposer


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('generated'))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    args.output.mkdir(parents=True, exist_ok=True)
    histogram(args.output)
    age_figures(args.csv, args.output)
