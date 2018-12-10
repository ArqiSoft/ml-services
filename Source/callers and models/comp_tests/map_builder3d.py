import pandas as pd
import plotly
import plotly.graph_objs as go

dataframe = pd.read_csv(
    'C:\PycharmProjects\ml.services\Source\callers and models\comp_tests\latent_descs_with_2d.csv',header=None,
    sep=' ')
dataframe_act = dataframe[dataframe[455] == 1]
dataframe_inact = dataframe[dataframe[455] == 0]
x1 = dataframe_act.iloc[:,456]
y1 = dataframe_act.iloc[:,457]
z1 = dataframe_act.iloc[:,458]
x2 = dataframe_inact.iloc[:,456]
y2 = dataframe_inact.iloc[:,457]
z2 = dataframe_inact.iloc[:,458]

trace1 = go.Scatter3d(
    x=x1,
    y=y1,
    z=z1,
    mode='markers',
    marker=dict(
        color='rgb(255, 0, 0)',
        size=3,
        line=dict(
            color='rgba(255, 0, 0, 0.14)',
            width=0.5
        ),
        opacity=0.8
    )
)
trace2 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        color='rgb(0, 0, 255)',
        size=3,
        symbol='circle',
        line=dict(
            color='rgb(0, 0, 255)',
            width=1
        ),
        opacity=0.9
    )
)
data = [trace1, trace2]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    )
)

fig = go.Figure(data=data, layout=layout)
plotly.offline.plot(fig, filename='simple-3d-scatter.html')
