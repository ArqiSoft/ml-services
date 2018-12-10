import csv
import json
import pandas as pd

df = pd.read_csv('C:\PycharmProjects\ml.services\Source\sds_tools\learner\evolution\\000b0000-ac12-0242-5616-08d61d952e3a-result.csv')
df = df.iloc[:,3:259]
df = df.iloc[0:200]
print(df)
df = df.loc[:, (df != 0).any(axis=0)]
print(list(df.max()))
print(list(df.min()))
df = df.to_json(orient='values',path_or_buf='starters.json')

