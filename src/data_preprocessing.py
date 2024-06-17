from langchain_community.document_loaders import PyMuPDFLoader
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataPreprocessor:
    def __init__(self, pdf_url):
        self.pdf_url = pdf_url
        self.docs = None
        self.split_chunks = None

    def load_and_split(self):
        loader = PyMuPDFLoader(self.pdf_url)
        self.docs = loader.load()
        
        print(self.docs)

        def tiktoken_len(text):
            tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(text)
            return len(tokens)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=0,
            length_function=tiktoken_len
        )

        self.split_chunks = text_splitter.split_documents(self.docs)
        return self.split_chunks
