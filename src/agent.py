from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

class Agent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.prompt = ChatPromptTemplate.from_messages(
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
        self.agent = create_tool_calling_agent(llm, tools, self.prompt)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=tools, verbose=True)

    def execute(self, input_text):
        return self.agent_executor.invoke({"input": input_text})
