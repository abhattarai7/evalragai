from azure.ai.contentsafety import ContentSafetyClient
from azure.identity import DefaultAzureCredential
import ssl
from azure.core.pipeline.transport import RequestsTransport

import requests
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager


class TLS12Adapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        context = ssl.create_default_context()
        context.maximum_version = ssl.TLSVersion.TLSv1_2

        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=context,
            **pool_kwargs
        )

endpoint = "https://evalrag-content-safety.cognitiveservices.azure.com/"

credential = DefaultAzureCredential()

tls_context = ssl.create_default_context()
tls_context.maximum_version = ssl.TLSVersion.TLSv1_2

transport = RequestsTransport(
    connection_verify=tls_context
)

session = requests.Session()
session.mount("https://", TLS12Adapter())

transport = RequestsTransport(session=session)

client = ContentSafetyClient(
    endpoint=endpoint,
    credential=credential,
    transport=transport
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