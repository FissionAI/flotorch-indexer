import unittest
from testcontainers.opensearch import OpenSearchContainer

from storage.db.vector.open_search import OpenSearchClient  # Assuming this is the class you want to test

class TestOpenSearchClient(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.opensearch_container = OpenSearchContainer(security_enabled=True)
        cls.opensearch_container.start()
        local_client = cls.opensearch_container.get_client()
        local_client.index(index="test", body={"test": "test"})
        cls.opensearch_client = OpenSearchClient(
            host=cls.opensearch_container.get_container_host_ip(),
                                        port=cls.opensearch_container.get_exposed_port(9200),
                                        username='admin',
                                        password='admin',
                                        index='test',
                                        use_ssl=True,
                                        verify_certs=False
                                        )
        
    @classmethod
    def tearDownClass(cls):
        cls.opensearch_container.stop()


    def test_write(self):
        body = {
            "name": "John Doe",
            "age": 30
        }
        # create open search client object at the class level and use it in the test
        self.opensearch_client.write(body)
    
    def test_read(self):
        body = {
            "query": {
                "match_all": {}
            }
        }
        # create open search client object at the class level and use it in the test
        response = self.opensearch_client.search(body)
        print(response)

if __name__ == '__main__':
    unittest.main()