from openai import OpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://zippfake-4596-resource.openai.azure.com/openai/v1/"

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default"
)

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider
)

def generate_answer(question, search_results):
    context = "\n\n".join(
        result["content"] for result in search_results 
    )

    response = client.responses.create(
        model="gpt-5.4-mini-1",
        instructions=(
            "You are a document question-answering assistant. "
            "Answer using only the provided document context. "
            "If the context does not contain the answer, say "
            "'I could not find the answer in the document.' "
            "Do not invent information."
        ),
        input=f"Document context:\n{context}\n\nQuestion: {question}",
        max_output_tokens=500
    )

    return response.output_text