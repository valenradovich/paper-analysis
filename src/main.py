import argparse
import re
from src.config import *
from src.data_preprocessing import DataPreprocessor
from src.embedding import EmbeddingModel
from src.rag_chain import RAGChain
from src.tools import Tools
from src.agent import Agent
from langchain_openai import ChatOpenAI

def is_valid_paper_url(url):
    """Check if the URL is from a recognized paper source (e.g., arXiv, IEEE Xplore, SpringerLink, etc.)."""
    patterns = [
        r"https:\/\/arxiv\.org\/pdf\/\d{4}\.\d{4,5}",  # arXiv PDF URL pattern
        r"https:\/\/scholar\.google\.com\/scholar\?q=([^&]+)",  # Google Scholar URL pattern
        r"https:\/\/ieeexplore\.ieee\.org\/document\/\d+",  # IEEE Xplore document URL pattern
        r"https:\/\/dl\.acm\.org\/doi\/abs\/\d{4}\/\d{4}",  # ACM Digital Library DOI URL pattern
        r"https:\/\/link\.springer\.com\/article\/10\.\d{4}\/s12345-6789-0"  # SpringerLink article URL pattern
    ]
    return any(re.match(pattern, url) for pattern in patterns)

def main():
    # Prompt the user for the URL
    pdf_url = input("\nPlease enter the URL of the paper.\n\n Must be: \n* directly to the PDF.\n* arXiv, IEEE Xplore, ACM Digital Library, SpringerLink, or Google Scholar.\n\nURL: ")

    if not is_valid_paper_url(pdf_url):
        print(f"Error: The URL '{pdf_url}' is not a valid paper URL.")
        return
    
    print("\nProcessing the paper...")
    data_preprocessor = DataPreprocessor(pdf_url)
    split_chunks = data_preprocessor.load_and_split()

    embedding_model = EmbeddingModel()
    qdrant_vectorstore = embedding_model.create_vectorstore(split_chunks)
    print("Vectorstore created successfully.")
    
    rag_chain = RAGChain(qdrant_vectorstore)

    tools = Tools(rag_chain)
    structured_tools = tools.create_structured_tools()

    llm = ChatOpenAI(temperature=0.0)
    agent = Agent(llm, structured_tools)

    input_text = """Identify and answer the following questions based on the paper content and research:
    - What is the research problem the paper attempts to address? What is the motivation?
    - What are the claimed contributions and novelties of the paper?
    - How do the authors substantiate their claims (methodology, experiments, proofs etc.)?
    - What are the main conclusions and lessons learned?
    - Is the research problem significant and not artificial?
    - Are the claimed contributions really significant and novel compared to existing work?
    - Are the claims and arguments valid, or are there flaws in the methodology, proofs, experiments etc.?
    - What is the core of the research problem? Are there alternative approaches?
    - Are there different ways to substantiate or argue against the claims?
    - Can the results be strengthened or applied to other contexts?
    - What are the open problems raised that could lead to further research?

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
    """
    
    # Ensure input_text is a plain string
    if isinstance(input_text, str):
        output = agent.execute(input_text)
        print(output['output'])
    else:
        print("Error: The input text is not a valid string.")

if __name__ == "__main__":
    main()
