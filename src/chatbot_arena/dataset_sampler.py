from huggingface_hub import login
from datasets import load_dataset
import random


# Authenticate (use a token with "read" access to private repos)
#login(token="<hf_token>")


class QuestionSampler:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.mcq_ds = load_dataset("AstroMLab/araa-mcq-gemini-1.5-generated-v2-temp-0")['train']
        self.qa_ds = load_dataset("AstroMLab/araa-qa-gemini-1.5-generated-v2")['train']
        self.mcq_index = 0
        self.qa_index = 0
        self.toggle = True  # Start with MCQ
        self.shuffle()

    def shuffle(self):
        self.mcq_ids = list(range(len(self.mcq_ds)))
        self.qa_ids = list(range(len(self.qa_ds)))
        random.shuffle(self.mcq_ids)
        random.shuffle(self.qa_ids)

    def get_next_question(self):
        if self.toggle:
            question = self._get_mcq()
        else:
            question = self._get_qa()

        self.toggle = not self.toggle  # Alternate next time
        self.last_question = question
        return question

    def _get_mcq(self):
        idx = self.mcq_ids[self.mcq_index % len(self.mcq_ids)]
        self.mcq_index += 1
        item = self.mcq_ds[idx]
        return {
            "type": "mcq",
            "question": item["question"],
            "choices": {"A": item["A"], "B": item["B"], "C": item["C"], "D": item["D"]},
            "correct": item["correct"],
            "explanation": item["explanation"],
            "difficulty": item["difficulty"],
            "paper_id": item["paper_id"]
        }

    def _get_qa(self):
        idx = self.qa_ids[self.qa_index % len(self.qa_ids)]
        self.qa_index += 1
        item = self.qa_ds[idx]
        return {
            "type": "qa",
            "question": item["question"],
            "answer": item["answer"],
            "explanation": item["explanation"],
            "difficulty": item["difficulty"],
            "paper_id": item["paper_id"]
        }


if __name__ == "__main__":
    print("Testing alternating question sampler...\n")

    sampler = QuestionSampler(seed=123)

    for i in range(10):
        q = sampler.get_next_question()
        print(f"[{i+1}] Type: {q['type'].upper()}")
        print(f"Q: {q['question']}")
        if q['type'] == 'mcq':
            for k, v in q['choices'].items():
                print(f"   {k}: {v}")
        elif q['type'] == 'qa':
            print(f"A: {q['answer']}")
        print("-" * 60)