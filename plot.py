import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events

df = pd.read_parquet("./data/subjects.parquet")


fig = go.Figure(
    go.Icicle(
        labels=df["id"],
        parents=df["parent"],
        text=df["description"],
        # maxdepth=,
    )
)
fig = fig.update_layout(margin=dict(b=0, l=0, r=0))
