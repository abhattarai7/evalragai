from azure.ai.contentsafety import ContentSafetyClient
from azure.identity import DefaultAzureCredential

endpoint = "https://evalrag-content-safety.cognitiveservices.azure.com/"

credential = DefaultAzureCredential()

client = ContentSafetyClient(
    endpoint=endpoint,
    credential=credential
)

from azure.ai.contentsafety.models import AnalyzeTextOptions

def analyze_safety(answer):
    request = AnalyzeTextOptions(text=answer)

    response = client.analyze_text(request)

    return response.categories_analysis

if __name__ == "__main__":
    test_answer = "Employees receive 15 paid vacation days per year."

    results = analyze_safety(test_answer)

    for result in results:
        print(result.category, ":", result.severity)