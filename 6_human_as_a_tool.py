# Human as a tool
# Get the AI to ask the human for more information to help reach an answer

import os
from dotenv import load_dotenv

# load in the env variables
load_dotenv()

# get the env variable relevant and needed
os.getenv('OPENAI_API_KEY')

#import get_all_tool_names module from langchain agents
# these are all the tools we have access to for our agents
from langchain.agents import get_all_tool_names, load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI

# setup openAI
llm = OpenAI(temperature=0) # makes sure the data is consistent and not creative as we read data

# setup the tools we will be using to get the answer
# NOTE: Some tools use an LLM so we need to pass the LLM along with the tools to help as a dependecy
# as some have the LLM create code that are executable
tools = load_tools(["human"]) # the answer from get_all_tool_names

# agent setup
# different agent types (7)
# zero-shot-react-description - uses the react framework to know what tool to use based soley on the tools description
# i.e all the tools passed we have have descriptions and this looks at that to decide which one based on the section in the prompt
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# prompt we are trying to get up to date information from
prompt = "Name 3 knives Keith has"
agent.run(prompt)