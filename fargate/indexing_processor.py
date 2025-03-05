import json
from embedding.titanv2_embedding import TitanV2Embedding
from embedding.titanv1_embedding import TitanV1Embedding
from embedding.cohere_embedding import CohereEmbedding
from embedding.bge_large_embedding import BGELargeEmbedding, BGEM3Embedding, GTEQwen2Embedding
from fargate.base_task_processor import BaseFargateTaskProcessor
from logger.global_logger import get_logger
from reader.pdf_reader import PDFReader
from storage.db.dynamodb import DynamoDB
from storage.db.vector import OpenSearchClient
from indexing.indexing import Index
from config.config import Config
from config.env_config_provider import EnvConfigProvider
from storage.storage_provider_factory import StorageProviderFactory
from embedding.embedding_registry import embedding_registry
from chunking.chunking_provider_factory import ChunkingFactory



logger = get_logger()
env_config_provider = EnvConfigProvider()
config = Config(env_config_provider)

class IndexingProcessor(BaseFargateTaskProcessor):
    """
    Processor for indexing tasks in Fargate.
    """

    def process(self):
        logger.info("Starting indexing process.")
        try:
            exp_config_data = self.input_data

            logger.info(f"Into indexing processor. Processing event: {json.dumps(exp_config_data)}")

            index_id = exp_config_data.get("index_id")

            kb_data = exp_config_data.get("kb_data")
            storage = StorageProviderFactory.create_storage_provider(kb_data)
            kb_data_path = storage.get_path(kb_data)
            pdf_reader = PDFReader(storage)

            chunking = ChunkingFactory.create_chunker(
                exp_config_data.get("chunking_strategy"), 
                int(exp_config_data.get("chunk_size", exp_config_data.get("hierarchical_child_chunk_size"))), 
                int(exp_config_data.get("chunk_overlap", exp_config_data.get("hierarchical_chunk_overlap_percentage"))),
                int(exp_config_data.get("hierarchical_parent_chunk_size", 0))
            )

            embedding_class = embedding_registry.get_model(exp_config_data.get("embedding_model"))
            embedding = embedding_class(exp_config_data.get("embedding_model"), exp_config_data.get("aws_region"), int(exp_config_data.get("vector_dimension")))


            indexing = Index(pdf_reader, chunking, embedding)
            embeddings_list = indexing.index(kb_data_path)

            logger.info("Indexing completed")


            # adding total_index_embed_tokens to dynamo db table: experiment_table
            experiment_db = DynamoDB(config.get_experiment_table_name(), config.get_region()) 
            experiment_db.update(
                key={'id': exp_config_data.get("experiment_id")},
                data={'index_embed_tokens': embeddings_list.metadata.input_tokens }
            )

            logger.info("DB update completed")

            open_search_client = OpenSearchClient(
                config.get_opensearch_host(), 
                config.get_opensearch_port(),
                config.get_opensearch_username(), 
                config.get_opensearch_password(),
                index_id
            )
    
            bulk_data = []
            for embedding in embeddings_list.embeddings:
                bulk_data.append({"index": {"_index": index_id}})
                data = embedding.to_json()
                # data['_index'] = index_id
                data['execution_id'] = exp_config_data.get('execution_id')
                if exp_config_data.get("chunking_strategy").lower() == 'hierarchical':
                    data['parent_id'] = embedding.id
                bulk_data.append(data)

            bulK_result = open_search_client.write_bulk(body=bulk_data)

            if bulK_result['errors']:
                logger.error("Error during bulk indexing")
                logger.error(f"Item 1: {json.dumps(bulK_result['items'][0])}")
                raise Exception("Error during bulk indexing")
            
            logger.info("opensearch bulk write completed")
            output = {"status": "success", "message": "Indexing completed successfully."}
            self.send_task_success(output)
        except Exception as e:
            logger.error(f"Error during indexing process: {str(e)}")
            self.send_task_failure(str(e))