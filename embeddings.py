import ssl
import httpx2
from openai import OpenAI 
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

endpoint = "https://zippfake-4596-resource.openai.azure.com/openai/v1/"

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(),
    "https://ai.azure.com/.default"
)

tls_context = ssl.create_default_context()
tls_context.maximum_version = ssl.TLSVersion.TLSv1_2

http_client = httpx2.Client(verify=tls_context)

client = OpenAI(
    base_url=endpoint,
    api_key=token_provider,
    http_client=http_client
)

def embed_chunks(chunks):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )

    embeddings = []

    for item in response.data:
        embeddings.append(item.embedding)

    return embeddings

def embed_query(query):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )

    return response.data[0].embedding