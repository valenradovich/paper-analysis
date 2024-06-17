from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

class EmbeddingModel:
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name
        self.embedding_model = OpenAIEmbeddings(model=self.model_name)
        self.qdrant_vectorstore = None

    def create_vectorstore(self, split_chunks):
        self.qdrant_vectorstore = Qdrant.from_documents(
            split_chunks,
            self.embedding_model,
            location=":memory:",
            collection_name='paper_vectors'
        )
        return self.qdrant_vectorstore
