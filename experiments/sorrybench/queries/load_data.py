from datasets import load_dataset
import pandas as pd

# Load sorry-bench (public mirror; for the gated version use
# "sorry-bench/sorry-bench-202503" and run `huggingface-cli login` first)
ds = load_dataset("AIM-Harvard/sorrybench", split="train")
df = pd.DataFrame(ds)

# Keep only base prompt style (one row per unique question), first 24 categories
base = df[(df["prompt_style"] == "base") & (df["category"].astype(int) <= 24)].copy()

# Stratified random sample: 100 questions across 24 categories
# ~4 per category (96) + 4 extra drawn from random categories
SEED = 42
N = 100

sampled = base.groupby("category", group_keys=False).apply(
    lambda g: g.sample(n=min(4, len(g)), random_state=SEED),
    include_groups=False,
)
sampled = base.loc[sampled.index]
remaining = base.drop(sampled.index)
extra = remaining.sample(n=N - len(sampled), random_state=SEED)
sampled = pd.concat([sampled, extra]).sort_values("question_id").reset_index(drop=True)

print(f"Sampled {len(sampled)} questions from {sampled['category'].nunique()} categories")
print(f"Per-category counts:\n{sampled['category'].value_counts().sort_index()}")

# Save as queries.csv (extract first turn as the query)
sampled["query"] = sampled["turns"].apply(lambda t: t[0])
sampled.rename(columns={"question_id": "id"}, inplace=True)
sampled[["id", "category", "query"]].to_csv(
    "experiments/sorrybench/queries/queries.csv", index=False
)
print("Saved to experiments/sorrybench/queries.csv")
