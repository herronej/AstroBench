import pyarrow as pa
from PIL import Image
import pandas as pd
from datasets import Dataset, load_dataset

df = pd.read_json("Olympiad_MCQ.jsonl", lines=True)

records = []
for index, row in df.iterrows():

    image_list = []
    for image_filename in row['images']:
        with Image.open(image_filename) as im:
            image_list.append(im.tobytes())
    dict_record = {
        "question": row["question"],
        "A": row["A"],
        "B": row["A"],
        "C": row["A"],
        "D": row["A"],
        "E": row["A"],
        "answer": row["A"],
        "images": image_list,
    }
    records.append(dict_record)

table = pa.Table.from_pylist(records)

df = table.to_pandas()
dataset = Dataset.from_pandas(df)

dataset.push_to_hub("AstroMLab/USAAAO_MCQ")

# load dataset from huggingface
dataset = load_dataset("AstroMLab/USAAAO_MCQ", split='train')
