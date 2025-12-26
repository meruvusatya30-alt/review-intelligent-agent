import pandas as pd
from agents.embedding_agent import embed
from agents.topic_memory_agent import assign_topic
from agents.trend_agent import generate_trend

df = pd.read_csv("data/reviews.csv")
df["date"] = pd.to_datetime(df["date"])

df["embedding"] = df["review"].apply(embed)
df["topic"] = df.apply(
    lambda x: assign_topic(x["embedding"], x["review"]),
    axis=1
)

latest = df["date"].max()
df = df[df["date"] >= latest - pd.Timedelta(days=30)]

trend = generate_trend(df)
trend.to_csv("output/trend_report.csv")

print("✅ Autonomous Trend Report Generated")
