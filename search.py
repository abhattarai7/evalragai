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

    SimpleField(
        name="document_id",
        type=SearchFieldDataType.String,
        filterable=True
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

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search
)

def upload_chunks(chunks, embeddings, document_id):
    search_client = SearchClient(
        endpoint=search_endpoint,
        index_name=index_name,
        credential=credential
    )

    documents = []

    for i in range(len(chunks)):
        document = {
            "id": f"{document_id}-chunk-{i}",
            "document_id": document_id,
            "content": chunks[i],
            "contentVector": embeddings[i]
        }

        documents.append(document)

    result = search_client.upload_documents(documents=documents)

    return result

def search_chunks(query_embedding, document_id, top_k=3):
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
        filter=f"document_id eq '{document_id}'",
        select=["id", "content", "document_id"]
    )

    return list(results)

if __name__ == "__main__":
    existing_index = index_client.get_index(index_name)

    if not any(field.name == "document_id" for field in existing_index.fields):
        existing_index.fields.append(
            SimpleField(
                name="document_id",
                type=SearchFieldDataType.String,
                filterable=True
            )
        )
        index_client.create_or_update_index(existing_index)
        print("Index updated successfully.")
    else:
        print("document_id field already exists.")