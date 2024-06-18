import asyncio
from typing import Annotated
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool

class Tools:
    def __init__(self, rag_chain, first_chunk_content):
        self.rag_chain = rag_chain
        self.first_chunk_content = first_chunk_content

    def write_markdown_file(self, content: Annotated[str, "The string final content to write to the file."], filename: Annotated[str, "The filename to save the file as, must be the paper title. do NOT include .md on it."]):
        """Writes the given content as a markdown file to the local directory."""
        with open(f"documents/{filename}.md", "w") as f:
            f.write(content)

    async def retrieve_paper_title(self):
        """Returns the paper title to use later"""
        prompt = f"Based on the following content, what is the paper title? Provide only the title without any additional text:\n\n{self.first_chunk_content}"
        return await self.rag_chain.ainvoke({"question": f"{prompt}"})

    async def retrieve_information(self, query: Annotated[str, "query to ask the retrieve information tool"], paper_title: Annotated[str, 'title of the paper']):
        """Use Retrieval Augmented Generation to retrieve information about the paper."""
        return await self.rag_chain.ainvoke({"question": query})

    def create_structured_tools(self):
        retrieve_paper_title = StructuredTool.from_function(
            coroutine=self.retrieve_paper_title, # changing func to coroutine
            name='RetrievePaperTitle',
            description="retrieve the paper's title"
        )

        retrieve_information = StructuredTool.from_function(
            coroutine=self.retrieve_information, # changing func to coroutine
            name='RetrieveInformation',
            description='search for information about the given paper using Retrieval Augmented Generation.'
        )

        write_content = StructuredTool.from_function(
            func=self.write_markdown_file,
            name="WriteContent",
            description="write the final content as a markdown file to the local directory."
        )

        return [retrieve_paper_title, retrieve_information, write_content]
