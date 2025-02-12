import os
from opensearchpy import OpenSearch
from storage.db.vector.vector_storage import VectorStorage
from typing import List

"""
This class is responsible for storing the data in the OpenSearch.
"""

class OpenSearchClient(VectorStorage):
    def __init__(self, host, port, username, password, index, use_ssl=True, verify_certs=True, ssl_assert_hostname=False, ssl_show_warn=False):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.index = index
        
        self.client = OpenSearch(
            hosts=[{'host': self.host, 'port': self.port}],
            http_auth=(self.username, self.password),
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ssl_assert_hostname=ssl_assert_hostname,
            ssl_show_warn=ssl_show_warn,
        )
    
    def write(self, body):
        return self.client.index(index=self.index, body=body)
    
    def read(self, body):
        return self.search(self.index, body)
    
    def write_bulk(self, body: List[dict]):
        return self.client.bulk(body=body)

    def search(self, body):
        response = self.client.search(index=self.index, body=body)
        return [hit['_source'] for hit in response['hits']['hits']]
    
    def embed_query(self, query_vector: List[float], knn: int):
        vector_field = next((field for field, props in 
                            self.client.indices.get_mapping(index=self.index)[self.index]['mappings']['properties'].items() 
                            if 'type' in props and props['type'] == 'knn_vector'), None)
        return {
            "size": knn,
            "query": {
                "knn": {
                    vector_field: {
                        "vector": query_vector,
                        "k": knn
                    }
                }
            },
            "_source": True,
            "fields": ["text", "parent_id"]
        }