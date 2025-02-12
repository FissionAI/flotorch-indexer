from storage.db.db_storage import DBStorage
from abc import ABC, abstractmethod

class VectorStorage(DBStorage, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @abstractmethod
    def search(self, body):
        pass

    @abstractmethod
    def embed_query(self, embedding, knn):
        pass
    
