import pandas as pd
import os
import shutil
import kagglehub
from pathlib import Path
import re
import json
from document_process import DocumentProcessor


os.makedirs("data", exist_ok=True)

queastion_file_path = Path("data/Questions.csv") 
answer_file_path = Path("data/Answers.csv") 
tags_file_path = Path("data/Tags.csv") 

if not queastion_file_path.exists() or not answer_file_path.exists() or not tags_file_path.exists():
    print("This 3 file is not exists in data foler  ")
    print("Downloading dataset...")
    path = kagglehub.dataset_download("stackoverflow/stacksample")
    print("Path to dataset files:", path)
    for file in os.listdir(path=path):
        if file.endswith(".csv"):
            shutil.copy(os.path.join(path, file), "data")

    print("Files copied to data/")




# script_location = os.path.dirname(os.path.abspath(__file__))
# print(script_location)





questions = pd.read_csv("data/Questions.csv", encoding="latin1", low_memory=False)
answers = pd.read_csv("data/Answers.csv", encoding="latin1", low_memory=False)
tags = pd.read_csv("data/Tags.csv", encoding="latin1", low_memory=False)

# print(queastions.head())
# print(answer.head())

# df_q = pd.DataFrame(queastions)
# has_q_duplicate = df_q["OwnerUserId"].duplicated().any()
# has_q_id_duplicate = df_q["Id"].duplicated().any()
# print(has_q_duplicate)
# print(has_q_id_duplicate)
# print(df_q)


# has_parent_id_duplplicate = df_a["Id"].duplicated().any()
# print(has_parent_id_duplplicate)

# Sort answers by score descending
answers_sorted = answers.sort_values("Score", ascending=False)

# Keep highest score answer per question
best_answers = answers_sorted.drop_duplicates("ParentId")   

# answer_counts = answers.groupby("ParentId").size()

# questions["answer_count"] = questions["Id"].map(answer_counts)

# print(questions["answer_count"])

qa = questions.merge(
    best_answers,
    left_on="Id",
    right_on="ParentId",
    how="inner",
    suffixes=("_question", "_answer")
)

# has_q_id_duplicate = qa["ParentId"].duplicated().any()
# print(has_q_id_duplicate)

# print(qa.head())
# print(qa.columns)

preprocessor = DocumentProcessor(qa_df=qa, tag_df=tags)

documents = preprocessor.process_all()

preprocessor.get_statistics()

preprocessor.save_documents("data/processed_documents.jsonl")

print("\n📄 Example Document:")

print(json.dumps(documents[0], indent=2))