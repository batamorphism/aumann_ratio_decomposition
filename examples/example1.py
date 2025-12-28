import re

import pandas as pd

from aumann_ratio_decomposition import (
    AumannShapleyRatioDecomposer,
    SDConfig,
    SubgroupLooper,
)

# csv読み込み
path = "csv/Kaggle.csv"       # ipynb と同じ階層にある csv フォルダ
df = pd.read_csv(path, sep=';', thousands=',', parse_dates=['Issue Date'], dayfirst=True)
df.head()

def make_safe_colnames(cols):
    mapping = {}
    for col in cols:
        safe = re.sub(r"[^A-Za-z0-9_]", "_", col)   # 特殊文字 → _
        safe = re.sub(r"_+", "_", safe).strip("_")  # _ をまとめて整形
        if safe and safe[0].isdigit():              # 先頭が数字？
            safe = "_" + safe
        mapping[col] = safe
    return mapping

rename_map = make_safe_colnames(df.columns)
df = df.rename(columns=rename_map)

# 型の整備
clean = (df['Premium'].astype(str)
                      .str.replace(r'[^\d\.\-]', '', regex=True))   # 数字・小数点・マイナス以外除去
df['Premium'] = pd.to_numeric(clean, errors='coerce') 

clean = (df['BENEFIT'].astype(str)
                      .str.replace(r'[^\d\.\-]', '', regex=True))   # 数字・小数点・マイナス以外除去
df['BENEFIT'] = pd.to_numeric(clean, errors='coerce') 

df['Issue_Date'] = pd.to_datetime(df['Issue_Date'],
                                  format='%b-%y',   # 月-年表記
                                  errors='coerce')  # うまく読めない行は NaT

df[f"Issue_year"]  = df['Issue_Date'].dt.year.astype("Int16")
df[f"Issue_month"] = df['Issue_Date'].dt.month.astype("Int8")
df[f"Issue_day"]   = df['Issue_Date'].dt.dayofyear.astype("Int16")
df.drop(columns=['Issue_Date'], inplace=True)  # 不要な列は削除

df['ENTRY_AGE_RANK'] = df['ENTRY_AGE'].astype("Int8") // 10 * 10  # 年齢を10歳刻みに

# NAを平均で埋める
df['den'] = df['Premium'].fillna(df['Premium'].mean())
df['den'] = df['den'].abs()  # Premium の絶対値を分母とする
# Lapse 件数を分子とする
df['numtest'] = df['POLICY_STATUS'].apply(lambda x: 1 if x == 'Lapse' else 0)
df['num'] = df['numtest'] * df['den']

df_f = df[df['SEX'] == 'F']
df_m = df[df['SEX'] == 'M']

df_f.drop(columns=['SEX'], inplace=True)
df_m.drop(columns=['SEX'], inplace=True)

shapley_my_data = AumannShapleyRatioDecomposer(
        df_before=df_f,
        df_after =df_m,
        mode="group", keys=["POLICY_TYPE_3", "Policy_Year"], den_col="den", num_col="num"
    )

shapley_my_data.plot_ratio_with_shapley_stacked(
    x_key="Policy_Year",
    decompose_keys=["ENTRY_AGE_RANK"],
)

ret = shapley_my_data.result()
ret

# 感応度を乗算
ret_prepped = ret[['Policy_Year_Decimal', 'CHANNEL1', 'CHANNEL2', 'CHANNEL3',
                   'ENTRY_AGE', 'POLICY_TYPE_3', 'PAYMENT_MODE', 'aumann_shapley', 'den_bef', 'den_aft']].copy()
ret_prepped['MVL_effect'] = ret_prepped['aumann_shapley'] * ret_prepped['Policy_Year_Decimal'] * (ret_prepped['den_bef'] + ret_prepped['den_aft'])/2
ret_prepped = ret_prepped.drop(columns=['den_bef', 'den_aft', 'aumann_shapley'])


# ret はあなたの元データフレーム
cfg = SDConfig(
    target_col='MVL_effect',
    use_abs_std_qf_numeric=True,
    depth=3,
    alpha_start=0.1,
    alpha_decay=0.3,
    min_support=1000,
    ratio_threshold=0.1,
    max_loops=None,
    intervals_only=False,
    nbins = 5,
)

runner = SubgroupLooper(ret_prepped, cfg)
df_summary = runner.run()
print(df_summary)

runner.plot(df_summary, wrap_width=40, fig_width=12, font_size=16)

