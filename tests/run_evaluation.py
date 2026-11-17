import json
import hashlib
from pathlib import Path

from embeddings import embed_query
from search import search_chunks
from generation import generate_answer

from evaluation import evaluate_correctness

handbook_path = Path("test_data/evalrag_employee_handbook.pdf")

pdf_bytes = handbook_path.read_bytes()
document_id = hashlib.sha256(pdf_bytes).hexdigest()

print("Testing document:", document_id[:12])

dataset_Path = Path(__file__).parent / "eval_dataset.json"

with open(dataset_Path, "r") as file:
    test_cases = json.load(file)

print("Number of test cases:", len(test_cases))

def check_correctness(answer, expected_answer, answerable):
    if not answerable:
        return "I could not find the answer in the document." in answer

    return expected_answer.lower() in answer.lower()

correct_count = 0 

semantic_correct_count = 0

for test in test_cases:
    question = test["question"]

    query_embedding = embed_query(question)
    search_results = search_chunks(query_embedding, document_id)
    answer, usage = generate_answer(question, search_results)

    is_correct = check_correctness(
        answer,
        test["expected_answer"],
        test["answerable"]
    )

    if test["answerable"]:
        semantic_result = evaluate_correctness(
            question,
            answer,
            test["expected_answer"]
        )
    else:
        semantic_result = (
            "CORRECT" if is_correct else "INCORRECT"
        )

    print("\nQuestion:", question)
    print("AI Answer:", answer)
    print("Expected Answer:", test["expected_answer"])
    print("Correct:", is_correct)

    if is_correct:
        correct_count += 1

    if semantic_result == "CORRECT":
        semantic_correct_count += 1

total_questions = len(test_cases)
pass_rate = (correct_count / total_questions) * 100
semantic_pass_rate = (semantic_correct_count / total_questions) * 100

print("\nEvaluation Summary")
print("Correct answers:", correct_count, "/", total_questions)
print("Exact-phrase pass rate:", round(pass_rate, 2), "%")

print("Semantic correct answers:", semantic_correct_count, "/", total_questions)
print("Semantic correctness rate:", round(semantic_pass_rate, 2), "%")
