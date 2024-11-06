from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from dataclasses import dataclass, field

from vectorizer import Vectorizer

class RAGAgent:
    def __init__(
        self,
        vectorizer: Vectorizer,
        tools: list,
        prompt:str,
        model: str = "gpt-4o",
        temperature: float = 0,
        verbose: bool = True,
        max_iterations: int = 3,
        chat_history: list = []
    ):
        
        self.llm = ChatOpenAI(model=model, temperature=temperature)

        self.vectorizer = vectorizer

        self.chat_history = chat_history
        self.tools = tools                 

        self.prompt = prompt
        
        # Create agent
        llm_with_tools = self.llm.bind_tools(self.tools)
        agent = (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | self.prompt
            | llm_with_tools
            | OpenAIToolsAgentOutputParser()
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=verbose,
            max_iterations=max_iterations,
            early_stopping_method="generate"
        )
    
    def query(self, user_input: str) -> str:
        """
        Query the agent and maintain chat history.
        
        Args:
            user_input: The user's question or input
        
        Returns:
            str: The agent's response
        """
        try:
            # Execute the agent
            result = self.agent_executor.invoke(
                {
                    "input": user_input,
                    "chat_history": self.chat_history
                }
            )
            
            # Update chat history
            self.chat_history.extend([
                HumanMessage(content=user_input),
                AIMessage(content=result["output"]),
            ])
            
            return result["output"]
        except Exception as e:
            return "Something went wrong!"
        
    def get_chat_history(self) -> list:
        """
        Get the current chat history.
        
        Returns:
            list: The current chat history
        """
        return self.chat_history
    
    def clear_chat_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []