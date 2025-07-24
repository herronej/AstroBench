import sqlite3
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from GPro.preference import ProbitPreferenceGP
import random
import os
import time

HUMAN_DB_PATH = 'votes.db'
OUTPUT_DIR = 'gppl_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_votes():
    conn = sqlite3.connect(HUMAN_DB_PATH)
    df = pd.read_sql("SELECT * FROM votes", conn)
    conn.close()
    return df

def extract_preference_data(df):
    texts = []
    pairs = []
    text_to_model = []

    for _, row in df.iterrows():
        vote = row["vote"].strip().lower()
        a, b = row["model_a"], row["model_b"]
        resp_a, resp_b = row["model_a_response"], row["model_b_response"]

        if vote in ("model a", "model b"):
            idx_a, idx_b = len(texts), len(texts) + 1
            if vote == "model a":
                pairs.append([idx_a, idx_b])
            else:
                pairs.append([idx_b, idx_a])
            texts.extend([resp_a, resp_b])
            text_to_model.extend([a, b])

    return texts, np.array(pairs), text_to_model

def run_gppl(texts, pairs):
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(texts).toarray()
    gpr = ProbitPreferenceGP()
    gpr.fit(X, pairs)
    scores = gpr.predict(X)
    return scores

def aggregate_model_scores(scores, text_to_model):
    model_scores = defaultdict(list)
    for score, model in zip(scores, text_to_model):
        model_scores[model].append(score)
    avg_scores = {model: float(np.mean(vals)) for model, vals in model_scores.items()}
    return dict(sorted(avg_scores.items(), key=lambda x: x[1], reverse=True))

def print_ranking(scores, title):
    print(f"\n=== Model Ranking by GPPL ({title}) ===")
    for rank, (model, score) in enumerate(scores.items(), 1):
        print(f"{rank}. {model}: {np.asarray(score).item():.4f}")

def save_rankings_to_csv(ranking, label):
    df = pd.DataFrame(ranking.items(), columns=['model', 'score'])
    df.to_csv(f"{OUTPUT_DIR}/gppl_scores_{label}.csv", index=False)

def plot_bootstrap_confidence_intervals(model_diffs):
    means, lowers, uppers = {}, {}, {}
    for model, diffs in model_diffs.items():
        if diffs:
            arr = np.array(diffs)
            means[model] = np.mean(arr)
            lowers[model] = np.percentile(arr, 2.5)
            uppers[model] = np.percentile(arr, 97.5)

    models = list(means.keys())
    mean_vals = [means[m] for m in models]
    lower_err = [means[m] - lowers[m] for m in models]
    upper_err = [uppers[m] - means[m] for m in models]

    plt.figure(figsize=(10, 6))
    plt.barh(models, mean_vals, xerr=[lower_err, upper_err], color='lightcoral', alpha=0.8)
    plt.axvline(0, color='gray', linestyle='--')
    plt.xlabel("Score Difference (Expert - Non-Expert)")
    plt.title("Bootstrapped GPPL Score Differences with 95% CI")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gppl_bootstrap_score_diff_ci.png", dpi=300)
    plt.show()

    # Save bootstrap data
    pd.DataFrame.from_dict(model_diffs, orient='index').T.to_csv(
        f"{OUTPUT_DIR}/gppl_bootstrap_raw_deltas.csv", index=False)

def bootstrap_score_differences(df, n_iterations=1000, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    models = sorted(set(df['model_a']) | set(df['model_b']))
    model_diffs = {model: [] for model in models}

    for _ in range(n_iterations):
        boot_df = df.sample(frac=1, replace=True)
        expert_df = boot_df[boot_df["astro_background"].str.lower() == "yes"]
        non_expert_df = boot_df[boot_df["astro_background"].str.lower() == "no"]

        if len(expert_df) < 5 or len(non_expert_df) < 5:
            continue

        try:
            texts_e, pairs_e, models_e = extract_preference_data(expert_df)
            scores_e = run_gppl(texts_e, pairs_e)
            ranking_e = aggregate_model_scores(scores_e, models_e)

            texts_n, pairs_n, models_n = extract_preference_data(non_expert_df)
            scores_n = run_gppl(texts_n, pairs_n)
            ranking_n = aggregate_model_scores(scores_n, models_n)
        except:
            continue

        for model in models:
            e = ranking_e.get(model, 0)
            n = ranking_n.get(model, 0)
            model_diffs[model].append(e - n)

    return model_diffs

# === Main Execution ===
df = load_votes()
expert_df = df[df["astro_background"].str.lower() == "yes"]
non_expert_df = df[df["astro_background"].str.lower() == "no"]

# Expert ranking
texts_e, pairs_e, models_e = extract_preference_data(expert_df)
scores_e = run_gppl(texts_e, pairs_e)
ranking_e = aggregate_model_scores(scores_e, models_e)
print_ranking(ranking_e, "Experts (Astro Background)")
save_rankings_to_csv(ranking_e, "experts")

# Non-expert ranking
texts_n, pairs_n, models_n = extract_preference_data(non_expert_df)
print("Starting GPPL on Non-Experts...")
start = time.time()
try:
    scores_n = run_gppl(texts_n, pairs_n)
except Exception as e:
    print("GPPL failed for Non-Experts:", e)
    scores_n = np.zeros(len(texts_n))
print("Finished GPPL for Non-Experts in", round(time.time() - start, 2), "seconds")

ranking_n = aggregate_model_scores(scores_n, models_n)
print_ranking(ranking_n, "Non-Experts")
save_rankings_to_csv(ranking_n, "nonexperts")

# Bootstrapped score deltas and figure
model_diffs = bootstrap_score_differences(df)
plot_bootstrap_confidence_intervals(model_diffs)