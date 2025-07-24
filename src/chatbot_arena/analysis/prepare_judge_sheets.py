import sqlite3
import pandas as pd
import random
import uuid
from datasets import load_dataset

# DB paths and vote column names
DB1_PATH = "votes.db"
DB2_PATH = "new_llm_votes.db"
HUMAN_COL = "vote"
LLM_COL = "llm_vote"
HUMAN_TABLE = "votes"
LLM_TABLE = "new_responses"

# Relevant comparison columns
CORE_COLUMNS = [
    "question_type", "question", "correct_answer", "explanation",
    "model_a", "model_a_response", "model_b", "model_b_response"
]

# === Step 1: Load MCQ Dataset for Option Lookup ===
print("Loading MCQ dataset...")
mcq_dataset = load_dataset("AstroMLab/araa-mcq-gemini-1.5-generated-v2-temp-0", split="train")

# Create lookup from question text to choices and correct letter
mcq_lookup = {
    item["question"].strip(): {
        "A": item["A"],
        "B": item["B"],
        "C": item["C"],
        "D": item["D"],
        "correct": item["correct"]
    }
    for item in mcq_dataset
}
print(f"Loaded {len(mcq_lookup)} MCQs with choices A–D and correct answers.")

# === Step 2: Load and Standardize Votes ===
def load_votes(db_path, vote_col, source_label, table_name):
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    all_required = [col for col in CORE_COLUMNS if col in df.columns] + [vote_col]
    df = df[all_required].copy()
    df.rename(columns={vote_col: "vote"}, inplace=True)
    df["source_id"] = [f"{source_label}_{uuid.uuid4().hex[:8]}" for _ in range(len(df))]
    df["source"] = source_label
    return df.sample(n=14, random_state=42).reset_index(drop=True)

# Load human and LLM votes
human_df = load_votes(DB1_PATH, HUMAN_COL, "human", HUMAN_TABLE)
llm_df = load_votes(DB2_PATH, LLM_COL, "llm", LLM_TABLE)

# Combine and shuffle
combined = pd.concat([human_df, llm_df], ignore_index=True)
combined = combined.sample(frac=1, random_state=123).reset_index(drop=True)

# === Step 3: Add Choices and Correct Answer ===
def get_choices(row):
    entry = mcq_lookup.get(row["question"].strip())
    if not entry:
        return None
    return {k: v for k, v in entry.items() if k in ["A", "B", "C", "D"]}

def format_choices(choices):
    if not choices:
        return ""
    return "\n".join([f"{k}. {v}" for k, v in choices.items()])

def get_correct_letter(row):
    entry = mcq_lookup.get(row["question"].strip())
    return entry["correct"] if entry and "correct" in entry else row["correct_answer"]

combined["choices"] = combined.apply(get_choices, axis=1)
combined["formatted_choices"] = combined["choices"].apply(format_choices)
combined["correct_answer"] = combined.apply(get_correct_letter, axis=1)

# === Step 4: Split into Judge Sheets ===
judge1 = pd.concat([
    combined[combined['source'] == 'human'].iloc[:7],
    combined[combined['source'] == 'llm'].iloc[:7]
]).sample(frac=1, random_state=1).reset_index(drop=True)

judge2 = pd.concat([
    combined[combined['source'] == 'human'].iloc[7:],
    combined[combined['source'] == 'llm'].iloc[7:]
]).sample(frac=1, random_state=2).reset_index(drop=True)

# === Step 5: Save Judge Sheets ===
def save_judge_file(df, filename):
    cols = [
        "question_type", "question", "formatted_choices", "correct_answer", "explanation",
        "model_a", "model_a_response", "model_b", "model_b_response", "vote"
    ]
    df[cols].to_excel(filename, index=False)

save_judge_file(judge1, "Judge1.xlsx")
save_judge_file(judge2, "Judge2.xlsx")

# === Step 6: Save Trace Mapping ===
trace_df = pd.concat([judge1, judge2], ignore_index=True)[["source_id", "source"]]
trace_df.to_csv("vote_source_mapping.csv", index=False)

print("Created: Judge1.xlsx, Judge2.xlsx")
print("Traceable mapping saved to vote_source_mapping.csv")