from azure.search.documents.models import VectorizedQuery
from azure.identity import DefaultAzureCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

search_endpoint = "https://evalrag-search.search.windows.net"
index_name = "evalrag-index"

credential = DefaultAzureCredential()

index_client = SearchIndexClient(
    endpoint=search_endpoint,
    credential=credential
)

fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True
    ),

    SearchableField(
        name="content",
        type=SearchFieldDataType.String
    ),

    SearchField(
        name="contentVector",
        type=SearchFieldDataType.Collection(
            SearchFieldDataType.Single
        ),
        searchable=True,
        vector_search_dimensions=1536,
        vector_search_profile_name="vector-profile"
    )
]

vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name="hnsw-config"
        )
    ],
    profiles=[
        VectorSearchProfile(
            name="vector-profile",
            algorithm_configuration_name="hnsw-config"
        )
    ]
)

try:
    index_client.delete_index(index_name)
    print("Deleting existing index:", index_name)
except Exception:
    pass

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search
)

result = index_client.create_or_update_index(index)

print("Created index:", result.name)

def upload_chunks(chunks, embeddings):
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=credential
    )

    documents = []

    for i in range(len(chunks)):
        document = {   
            "id": f"chunk-{i}",
            "content": chunks[i],
            "contentVector": embeddings[i]
        }

        documents.append(document)

    result = search_client.upload_documents(documents=documents)

    return result

def search_chunks(query_embedding, top_k=1):
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=credential
    )

    vector_query = VectorizedQuery(
        vector=query_embedding,
        k_nearest_neighbors=top_k,
        fields="contentVector"
    )

    results = search_client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "content"]
    )

    return list(results)