from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
import asyncio

class EmbeddingModel:
    def __init__(self, model_name="text-embedding-3-small"):
        self.model_name = model_name
        self.embedding_model = OpenAIEmbeddings(model=self.model_name)
        self.qdrant_vectorstore = None
        self.first_chunk_content = None

    async def create_vectorstore(self, split_chunks):
        texts = [doc.page_content for doc in split_chunks]
        metadatas = [{"chunk_index": i} for i, _ in enumerate(texts)]
        self.first_chunk_content = texts[0]  # Store the first chunk content

        self.qdrant_vectorstore = await Qdrant.afrom_texts(
            texts=texts,
            embedding=self.embedding_model,
            metadatas=metadatas,
            location=":memory:",
            collection_name='paper_vectors'
        )
        return self.qdrant_vectorstore

    def get_first_chunk(self):
        if self.first_chunk_content is None:
            raise ValueError("First chunk content is not available. Please create the vector store first.")

        return self.first_chunk_content

