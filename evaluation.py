from generation import client 

def evaluate_groundedness(question, answer, search_results):
    context = "\n\n".join(
        result["content"] for result in search_results
    )

    response = client.responses.create(
        model="gpt-5.4-mini-1",
        instructions=(
            "You are an AI evaluator. "
            "Evaluate the answer using only the provided document context. "
            "Return exactly one label: SUPPORTED, UNSUPPORTED, or INSUFFICIENT_INFORMATION. "
            "Return SUPPORTED if all factual claims in the answer are supported by the context. "
            "Return UNSUPPORTED if the answer contains factual claims not supported by the context. "
            "Return INSUFFICIENT_INFORMATION if the answer appropriately states "
            "that the requested information is unavailable in the context, "
            "without inventing facts. "
            "Return only the label."
        ),
        input=(
            f"Document context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            f"Answer:\n{answer}"
        ),
        max_output_tokens=300
    )

    return response.output_text.strip()

def evaluate_relevance(question, answer):
    response = client.responses.create(
        model="gpt-5.4-mini-1",
        instructions=(
            "You are an AI evaluator. "
            "Determine whether the answer addresses the user's question. "
            "Return exactly one label: RELEVANT, PARTIALLY_RELEVANT, or IRRELEVANT. "
            "Return RELEVANT if the answer directly addresses the question. "
            "Return PARTIALLY_RELEVANT if the answer addresses only part of the question. "
            "Return IRRELEVANT if the answer does not address the question. "
            "Return only the label."
        ),
        input=f"Question:\n{question}\n\nAnswer:\n{answer}",
        max_output_tokens=300
    )

    return response.output_text.strip()

if __name__ == "__main__":
    test_results = [
        {
            "content": "Full-time employees receive 15 paid vacation days per year."
        }
    ]

    vacation_question = "How many paid vacation days do employees receive?"
    correct_answer = "Employees receive 15 paid vacation days."
    incorrect_answer = "Employees receive 25 paid vacation days."

    print("Correct answer:", evaluate_groundedness(vacation_question, correct_answer, test_results))
    print("Incorrect answer:", evaluate_groundedness(vacation_question, incorrect_answer, test_results))
    
    salary_question = "What is the CEO's annual salary?"
    refusal_answer = "I could not find the answer in the document."

    print(
        "Refusal:",
        evaluate_groundedness(salary_question, refusal_answer, test_results)
    )

    print("Relevant:", evaluate_relevance(
        vacation_question,
        correct_answer
    ))

    print("Irrelevant:", evaluate_relevance(
        vacation_question,
        "Employees must use the corporate VPN when working remotely."
    ))

    print("Partially relevant:", evaluate_relevance(
        "How many vacation days do employees receive, and how many can carry over?",
        "Employees receive 15 paid vacation days."
    ))