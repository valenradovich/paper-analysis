# %% [markdown]
# ## this notebook was developed on Google Colab
# 

# %% [markdown]
# ### dependencies

# %%
import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

# %% [markdown]
# ### data collection and preprocessing

# %%
from langchain.document_loaders import PyMuPDFLoader

extending_context_window_llama_3 = "https://arxiv.org/pdf/2404.19553"
attention_is_all_you_need = "https://arxiv.org/pdf/1706.03762"

docs = PyMuPDFLoader(extending_context_window_llama_3).load()

# %%
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,
    chunk_overlap = 0,
    length_function = tiktoken_len,
)

split_chunks = text_splitter.split_documents(docs)

# %%
len(split_chunks)

# %% [markdown]
# ### embeddings and vectordb

# %%
from langchain_openai.embeddings import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# %%
from langchain_community.vectorstores import Qdrant

qdrant_vectorstore = Qdrant.from_documents(
    split_chunks,
    embedding_model,
    location=":memory:",
    collection_name='ExtendingLlama-3’sContext',
)

# %%
qdrant_retriever = qdrant_vectorstore.as_retriever()

# %% [markdown]
# ### rag chain

# %%
from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = """
CONTEXT:
{context}

QUERY:
{question}

You are a helpful assistant. Use the available context to answer the question. If you can't answer the question, say you don't know.
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# %%
from langchain_openai import ChatOpenAI

openai_chat_model = ChatOpenAI(model="gpt-3.5-turbo")

# %%
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser

rag_chain = (
    {"context": itemgetter("question") | qdrant_retriever, "question": itemgetter("question")}
    | rag_prompt | openai_chat_model | StrOutputParser()
)

# %%
rag_chain.invoke({"question" : "What does the 'context' in 'long context' refer to?"})

# %%
paper_title = rag_chain.invoke({"question" : "Retrieve the paper title as string without any adittional word."})
paper_title

# %% [markdown]
# ### tools

# %%
from typing import Annotated

def write_markdown_file(content: Annotated[str, "The string final content to write to the file."], filename: Annotated[str, "The filename to save the file as, must be the paper title. do NOT include .md on it."]):
  """Writes the given content as a markdown file to the local directory.

  Args:
    content: The string content to write to the file.
    filename: The filename to save the file as, must be the paper title, do NOT include .md.
  """
  with open(f"documents/{filename}.md", "w") as f:
    f.write(content)

# %%
def retrieve_paper_title():
  """Returns the paper title to use later"""
  return rag_chain.invoke({"question" : "Retrieve the paper title as string without any adittional word."})

# %%
def retrieve_information(query: Annotated[str, "query to ask the retrieve information tool"], paper_title: Annotated[str, 'title of the paper']):
    f"""Use Retrieval Augmented Generation to retrieve information about {paper_title} paper."""
    return rag_chain.invoke({"question" : query})

# %%
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool

retrieve_paper_title = StructuredTool.from_function(
                func=retrieve_paper_title,
                name='RetrievePaperTitle',
                description="retrieve the paper's title")

retrieve_information = StructuredTool.from_function(
                func= retrieve_information,
                name= 'RetrieveInformation',
                description= 'search for information about the given paper using Retrieval Augmented Generation.',)

write_content = StructuredTool.from_function(
                func=write_markdown_file,
                name="WriteContent",
                description="write the final content as a markdown file to the local directory.",)


tools = [retrieve_paper_title, retrieve_information, write_content]

# %%
llm = ChatOpenAI(temperature=0.0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Make sure to use the retrieve_information tool for information.",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

# %%
from langchain.agents import AgentExecutor, create_tool_calling_agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)#, handle_parsing_errors=True)

# %%
output = agent_executor.invoke(
    {
      "input": """Identify and answer the following questions based on the paper content and reasearch:
        -What is the research problem the paper attempts to address? What is the motivation?
        -What are the claimed contributions and novelties of the paper?
        -How do the authors substantiate their claims (methodology, experiments, proofs etc.)?
        -What are the main conclusions and lessons learned?
        -Is the research problem significant and not artificial?
        -Are the claimed contributions really significant and novel compared to existing work?
        -Are the claims and arguments valid, or are there flaws in the methodology, proofs, experiments etc.?
        -What is the core of the research problem? Are there alternative approaches?
        -Are there different ways to substantiate or argue against the claims?
        -Can the results be strengthened or applied to other contexts?
        -What are the open problems raised that could lead to further research?

        Then, based on the answers obtained, use and complete the following template to write a review of the paper:

        ## ReplaceThisWithPaperTitle

        ### Comprehension:
        \n- **Research Problem and Motivation:**
        \n- **Claimed Contributions and Novelties:**
        \n- **Substantiation of Claims:**
        \n- **Main Conclusions and Lessons Learned:**

        ### Evaluation:
        \n- **Significance of Research Problem:**
        \n- **Significance and Novelty of Contributions:**
        \n- **Validity of Claims and Arguments:**

        ### Synthesis:
        \n- **Core Research Problem:**
        \n- **Alternative Approaches and Substantiation:**
        \n- **Strengthening and Application of Results:**
        \n- **Open Problems for Further Research:**

        The review should be written in markdown format and saved as a file with the paper title as the filename.
        """,
    }
)

# %%
print(output['output'])


