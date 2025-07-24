import pandas as pd
import sqlite3
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === Load expert judgments ===
judge1 = pd.read_excel("Judge1.xlsx")
judge2 = pd.read_excel("Judge2.xlsx")
expert_df = pd.concat([judge1, judge2], ignore_index=True)
expert_df["question"] = expert_df["question"].str.strip()

# === Infer source using presence in o4-mini_votes.db ===
conn = sqlite3.connect("o4-mini_votes.db")
llm_judged = pd.read_sql("SELECT question FROM votes", conn)
conn.close()
llm_judged["question"] = llm_judged["question"].str.strip()
human_questions = set(llm_judged["question"])

def infer_source(row):
    return "human" if row["question"] in human_questions else "llm"

expert_df["source"] = expert_df.apply(infer_source, axis=1)

# === Helper functions ===
def compare_votes(df, col1, col2):
    agree = df[col1].str.lower().str.strip() == df[col2].str.lower().str.strip()
    return agree.sum(), len(df), agree

def print_agreement(df, label):
    agree, total, _ = compare_votes(df, "vote", "judge_vote")
    print(f"{label} Agreement: {agree}/{total} = {agree / total:.2%}")

def print_question_type_breakdown(df, label):
    grouped = df.groupby("question_type")
    print(f"\n{label} Agreement by Question Type:")
    for qt, group in grouped:
        agree, total, _ = compare_votes(group, "vote", "judge_vote")
        print(f" - {qt}: {agree}/{total} = {agree / total:.2%}")

def show_confusion(df, label):
    y_true = df["vote"].str.lower().str.strip()
    y_pred = df["judge_vote"].str.lower().str.strip()
    labels = sorted(set(y_true.dropna()) | set(y_pred.dropna()))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(f"\n Confusion Matrix: {label}")
    print(df_cm)

    plt.figure(figsize=(6, 5))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {label}")
    plt.ylabel("Human/LLM Vote")
    plt.xlabel("Expert Judgment")
    plt.tight_layout()
    path = f"confusion_matrix_{label.replace(' ', '_').lower()}.png"
    plt.savefig(path)
    print(f" Saved: {path}")

# === Overall agreement stats ===
print_agreement(expert_df, "Overall Expert")
print_question_type_breakdown(expert_df, "Overall Expert")
show_confusion(expert_df, "Expert vs Original Vote")

# === Source-specific stats ===
for src in ["human", "llm"]:
    subset = expert_df[expert_df["source"] == src]
    print_agreement(subset, f"Expert vs {src.capitalize()} Vote")
    print_question_type_breakdown(subset, f"Expert vs {src.capitalize()}")
    show_confusion(subset, f"{src.capitalize()} Only")

# === Save expert disagreement cases ===
expert_df["agree"] = expert_df.apply(
    lambda r: str(r["vote"]).strip().lower() == str(r["judge_vote"]).strip().lower(),
    axis=1
)
expert_df[~expert_df["agree"]].to_csv("expert_disagreements.csv", index=False)
print("\nSaved: expert_disagreements.csv")

# === Compare full LLM-as-Judge votes ===
print("\n Loading full LLM-as-Judge results from o4-mini_votes.db")
conn = sqlite3.connect("o4-mini_votes.db")
llm_df = pd.read_sql("SELECT * FROM votes", conn)
conn.close()

llm_df = llm_df[["question", "vote", "llm_vote"]].dropna()
llm_df["question"] = llm_df["question"].str.strip()
llm_df["vote"] = llm_df["vote"].str.strip().str.lower()
llm_df["llm_vote"] = llm_df["llm_vote"].str.strip().str.lower()
llm_df["agree"] = llm_df["vote"] == llm_df["llm_vote"]

agree_total = llm_df["agree"].sum()
print(f"\nLLM-as-Judge agreement with Human Votes: {agree_total}/{len(llm_df)} = {agree_total / len(llm_df):.2%}")

# === Cross-check 15 human-voted items reviewed by expert and o4-mini ===
print("\nCross-checking expert-judged human-voted questions with LLM-as-Judge...")
human_judged = expert_df[expert_df["source"] == "human"]
merged = pd.merge(human_judged, llm_df, on="question", suffixes=("_expert", "_llm"))

merged["llm_matches_expert"] = merged["llm_vote"] == merged["judge_vote"].str.strip().str.lower()
merged["llm_matches_human"] = merged["llm_vote"] == merged["vote_expert"].str.strip().str.lower()

match_expert = merged["llm_matches_expert"].sum()
match_human = merged["llm_matches_human"].sum()

print(f" - LLM-as-Judge agreement with expert: {match_expert}/{len(merged)} = {match_expert / len(merged):.2%}")
print(f" - LLM-as-Judge agreement with original human: {match_human}/{len(merged)} = {match_human / len(merged):.2%}")

merged.to_csv("crosscheck_human_llm_expert.csv", index=False)
print("Saved: crosscheck_human_llm_expert.csv")