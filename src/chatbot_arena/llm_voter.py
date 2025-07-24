import sqlite3
from datetime import datetime
import os
import requests
import json

DB_PATH = "new_responses.db"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def send_prompt_to_chatgpt(prompt, api_key):
    url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }
    data = {
        'model': 'o4-mini',
        'messages': [
            {'role': 'user', 'content': prompt}
        ]
    }

    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        raise Exception(f"Failed to fetch response: {response.text}")

def add_llm_vote_column():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("ALTER TABLE new_responses ADD COLUMN llm_vote TEXT;")
            print("Added 'llm_vote' column.")
        except sqlite3.OperationalError:
            print("'llm_vote' column already exists.")

def get_llm_vote(question, response_a, response_b, correct_answer, explanation):
    prompt = f"""
You are a scientific evaluator judging two anonymous responses to an astrophysics question.

Please analyze the following and return your judgment in JSON format.

### Input

Question:
{question}

Model A Response:
{response_a}

Model B Response:
{response_b}

Correct Answer: {correct_answer}
Explanation: {explanation}

### Instructions

Decide which model answered better. Choose exactly one of the following values:

- "Model A"
- "Model B"
- "Tie"
- "Both Wrong"

### Output format

Respond **only** with a valid JSON object using this structure:

{{
  "vote": "Model A"  // or "Model B", "Tie", or "Both Wrong"
}}

Do not include any additional commentary outside the JSON block.
"""
    response = send_prompt_to_chatgpt(prompt, OPENAI_API_KEY)
    try:
        parsed = json.loads(response)
        return parsed["vote"]
    except Exception as e:
        raise Exception(f"Failed to parse LLM JSON vote: {e}\nRaw response:\n{response}")


def annotate_llm_votes(test_one=False):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        rows = cursor.execute("""
            SELECT id, question, model_a_response, model_b_response, correct_answer, explanation, llm_vote 
            FROM new_responses
        """).fetchall()

        for row in rows:
            vote_id, question, resp_a, resp_b, correct, expl, existing_llm_vote = row
            if existing_llm_vote:
                continue
            try:
                print(f"\n[Processing vote ID {vote_id}]")
                llm_vote = get_llm_vote(question, resp_a, resp_b, correct, expl)
                print(f"  ➤ LLM Vote: {llm_vote}")
                cursor.execute("UPDATE new_responses SET llm_vote = ? WHERE id = ?", (llm_vote, vote_id))
                conn.commit()
                if test_one:
                    break  # exit after 1 row
            except Exception as e:
                print(f"  ✗ Error processing row {vote_id}: {e}")

if __name__ == "__main__":
    add_llm_vote_column()
    annotate_llm_votes()  # Change to False when ready
    print("Finished annotating LLM votes.")