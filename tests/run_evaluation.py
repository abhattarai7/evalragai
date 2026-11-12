import json
from pathlib import Path

from embeddings import embed_query
from search import search_chunks
from generation import generate_answer

dataset_Path = Path(__file__).parent / "eval_dataset.json"

with open(dataset_Path, "r") as file:
    test_cases = json.load(file)

print("Number of test cases:", len(test_cases))

def check_correctness(answer, expected_answer, answerable):
    if not answerable:
        return "I could not find the answer in the document." in answer

    return expected_answer.lower() in answer.lower()

for test in test_cases:
    question = test["question"]

    query_embedding = embed_query(question)
    search_results = search_chunks(query_embedding)
    answer, usage = generate_answer(question, search_results)

    print("\nQuestion:", question)
    print("AI Answer:", answer)
    print("Expected Answer:", test["expected_answer"])

