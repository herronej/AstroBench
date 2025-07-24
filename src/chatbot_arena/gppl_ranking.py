import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
from GPro.preference import ProbitPreferenceGP

# Paths to the databases
HUMAN_DB_PATH = 'votes.db'
LLM_DB_PATH = 'new_llm_votes.db'

def load_votes(db_path, table_name, vote_col, is_llm=False):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    if is_llm:
        df["astro_background"] = "No"
        df["vote"] = df[vote_col]
    return df

def extract_preference_data(df):
    texts = []
    pairs = []
    text_to_model = []

    for _, row in df.iterrows():
        vote = row["vote"].strip().lower()
        a, b = row["model_a"], row["model_b"]
        resp_a, resp_b = row["model_a_response"], row["model_b_response"]

        if vote == "model a":
            idx_a, idx_b = len(texts), len(texts) + 1
            pairs.append([idx_a, idx_b])
            texts.extend([resp_a, resp_b])
            text_to_model.extend([a, b])
        elif vote == "model b":
            idx_a, idx_b = len(texts), len(texts) + 1
            pairs.append([idx_b, idx_a])  # preference for B over A
            texts.extend([resp_a, resp_b])
            text_to_model.extend([a, b])
        else:
            # Skip "tie" and "both wrong"
            continue

    return texts, np.array(pairs), text_to_model

def run_gppl(texts, pairs):
    vectorizer = TfidfVectorizer(max_features=1000)
    X = vectorizer.fit_transform(texts).toarray()

    # Instantiate and fit the GPPL model
    gpr = ProbitPreferenceGP()
    gpr.fit(X, pairs)

    # Predict scores
    scores = gpr.predict(X)
    return scores

def aggregate_model_scores(scores, text_to_model):
    model_scores = defaultdict(list)
    for score, model in zip(scores, text_to_model):
        model_scores[model].append(score)
    avg_scores = {model: sum(vals)/len(vals) for model, vals in model_scores.items()}
    return dict(sorted(avg_scores.items(), key=lambda x: x[1], reverse=True))

# Load both datasets
human_df = load_votes(HUMAN_DB_PATH, "votes", "vote", is_llm=False)
llm_df = load_votes(LLM_DB_PATH, "new_responses", "llm_vote", is_llm=True)

# Process human votes
texts_human, pairs_human, text_to_model_human = extract_preference_data(human_df)
scores_human = run_gppl(texts_human, pairs_human)
final_scores_human = aggregate_model_scores(scores_human, text_to_model_human)

# Process LLM votes
texts_llm, pairs_llm, text_to_model_llm = extract_preference_data(llm_df)
scores_llm = run_gppl(texts_llm, pairs_llm)
final_scores_llm = aggregate_model_scores(scores_llm, text_to_model_llm)

# Display rankings
print("\n=== Model Ranking by GPPL (Human Votes) ===")
for rank, (model, score) in enumerate(final_scores_human.items(), 1):
    print(f"{rank}. {model}: {score.item():.4f}")

print("\n=== Model Ranking by GPPL (LLM Votes) ===")
for rank, (model, score) in enumerate(final_scores_llm.items(), 1):
    print(f"{rank}. {model}: {score.item():.4f}")
