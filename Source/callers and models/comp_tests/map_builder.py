
import matplotlib.pyplot as plt
import pandas as pd

dataframe = pd.read_csv(
    'C:\PycharmProjects\GT-Gen\maps_test\latent_descs_with_2d.csv',header=None,
    sep=' ')
dataframe_act = dataframe[dataframe[1] == 2]
dataframe_inact = dataframe[dataframe[1] == 1]
plt.scatter(dataframe_act.iloc[:,258], dataframe_act.iloc[:,259], alpha=1,c='r',s=1)
plt.scatter(dataframe_inact.iloc[:,258], dataframe_inact.iloc[:,259], alpha=1,s=1)

plt.savefig('123.png')