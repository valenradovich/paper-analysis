from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser

class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.rag_prompt = ChatPromptTemplate.from_template(
            """
            CONTEXT:
            {context}

            QUERY:
            {question}

            You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
            """
        )
        self.openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")
        self.rag_chain = (
            {"context": itemgetter("question") | self.vectorstore.as_retriever(), "question": itemgetter("question")}
            | self.rag_prompt | self.openai_chat_model | StrOutputParser()
        )

    # def invoke(self, question):
    #     return self.rag_chain.invoke({"question": question})

    def invoke(self, input_data):
        question = input_data.get("question", "")
        # Ensure question is a string
        if not isinstance(question, str):
            print(f"Expected a string for question, got {type(question)}")
            return {"error": "Invalid input type"}
        return self.rag_chain.invoke({"question": question})
