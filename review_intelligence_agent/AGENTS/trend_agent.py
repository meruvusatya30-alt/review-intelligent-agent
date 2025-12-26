import pandas as pd

def generate_trend(df):
    return (
        df.groupby(["topic", "date"])
        .size()
        .reset_index(name="count")
        .pivot(index="topic", columns="date", values="count")
        .fillna(0)
    )
