import sqlite3
import os
from dataset_sampler import QuestionSampler
from model_sampler import get_next_model_pair
from langchain_community.llms.sambanova import SambaStudio
from config import *

EXISTING_DB = "votes_copy.db"
NEW_DB = "new_responses.db"
NUM_SAMPLES = 100

os.environ["SAMBASTUDIO_URL"] = api_url
os.environ["SAMBASTUDIO_API_KEY"] = api_key

def create_model_instance(model_name):
    return SambaStudio(
        model_kwargs={
            "model": model_name,
            "max_tokens": 512,
            "temperature": 0.3,
            "top_p": 0.8,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "process_prompt": False
        }
    )

def load_existing_questions():
    with sqlite3.connect(EXISTING_DB) as conn:
        rows = conn.execute("SELECT question FROM votes").fetchall()
        return set(row[0].strip() for row in rows)

def init_new_db():
    with sqlite3.connect(NEW_DB) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS new_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_type TEXT,
            question TEXT,
            correct_answer TEXT,
            explanation TEXT,
            model_a TEXT,
            model_a_response TEXT,
            model_b TEXT,
            model_b_response TEXT
        );
        """)

def run_new_evaluations(test_one=False):
    existing_questions = load_existing_questions()
    sampler = QuestionSampler(seed=42)
    count = 0

    with sqlite3.connect(NEW_DB) as conn:
        while count < NUM_SAMPLES:
            question = sampler.get_next_question()
            if question["question"].strip() in existing_questions:
                continue

            model_a, model_b = get_next_model_pair()
            mod_a = create_model_instance(model_a)
            mod_b = create_model_instance(model_b)

            prompt = question["question"]
            if question["type"] == "mcq":
                for key, val in question["choices"].items():
                    prompt += f"\n({key}) {val}"

            response_a = mod_a.invoke(prompt)
            response_b = mod_b.invoke(prompt)

            conn.execute("""
            INSERT INTO new_responses (
                question_type, question, correct_answer, explanation,
                model_a, model_a_response, model_b, model_b_response
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                question["type"],
                question["question"],
                question.get("correct", question.get("answer", "")),
                question["explanation"],
                model_a,
                response_a,
                model_b,
                response_b
            ))
            conn.commit()
            count += 1
            print(f"[{count}/100] Logged new comparison")
            if test_one:
                break


    print("Completed generation of 100 new question-response sets.")

if __name__ == "__main__":
    init_new_db()
    run_new_evaluations()
