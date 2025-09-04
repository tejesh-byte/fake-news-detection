import pandas as pd

data = {
    "text": [
        "Breaking news about politics",
        "Celebrity gossip today",
        "Fake news story example",
        "Real news article"
    ],
    "label": [0, 0, 1, 0]  # 0 = real, 1 = fake
}

df = pd.DataFrame(data)
df.to_csv("data/fake_news.csv", index=False)
print("✅ Dummy dataset created at data/fake_news.csv")
