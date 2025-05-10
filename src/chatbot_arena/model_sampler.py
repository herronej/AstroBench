import sqlite3
import random
from collections import defaultdict

DB_PATH = "votes.db"

#arena model list
MODELS = [
    'astrollama-2-70b-chat_aic', 
    'astrollama-3-8b-chat_aic', 
    'astrollama-3-8b-chat_summary', 
    'astrollama-2-7b-chat_aic', 
    'AstroSage-8B', 
    'Qwen2-7B-Instruct', 
    'Meta-Llama-3-8B-Instruct', 
    'Meta-Llama-3-70B-Instruct',
    'Meta-Llama-3.1-405B-Instruct-FP8',
    'Meta-Llama-3.1-8B-Instruct'
]

def load_votes():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute("SELECT model_a, model_b FROM votes")
        return cursor.fetchall()

def get_models_voted(vote_records):
    models_seen = set()
    for a, b in vote_records:
        models_seen.update([a, b])
    return models_seen

def get_pairs_voted(vote_records):
    pairs = set()
    for a, b in vote_records:
        pair = tuple(sorted((a, b)))  # Order-independent
        pairs.add(pair)
    return pairs

def get_next_model_pair():
    vote_records = load_votes()
    models_seen = get_models_voted(vote_records)
    pairs_seen = get_pairs_voted(vote_records)

    # Priority 1: Ensure every model has been voted on at least once
    unseen_models = list(set(MODELS) - models_seen)
    if unseen_models:
        model_a = random.choice(unseen_models)
        model_b = random.choice([m for m in MODELS if m != model_a])
        return model_a, model_b

    # Priority 2: Cover all unique unordered pairs
    all_possible_pairs = set(
        tuple(sorted((a, b)))
        for i, a in enumerate(MODELS)
        for j, b in enumerate(MODELS)
        if i < j
    )
    remaining_pairs = list(all_possible_pairs - pairs_seen)
    if remaining_pairs:
        return random.choice(remaining_pairs)

    # Fallback: Everything covered, return random pair
    return random.sample(MODELS, 2)

if __name__ == "__main__":
    a, b = get_next_model_pair()
    print("Next model pair to vote on:")
    print(f"Model A: {a}")
    print(f"Model B: {b}")