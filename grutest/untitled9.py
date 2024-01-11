import pandas as pd

df = pd.read_csv("000001.csv", encoding='gbk')
df["日期"] = pd.to_datetime(df["日期"], format="%Y-%m-%d")
df.sort_values(by=["日期"], ascending=True)
df = df.sort_values(by=["日期"], ascending=True)
df.to_csv("sh.csv", index=False, sep=',')

