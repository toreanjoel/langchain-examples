# Action agents
# WHEN TO USE: Shorter tasks, you have a clear idea what to do, as info comes in to work towards the goal
# chain a series of calls and other tools based on user inputs to get to an answer
# Two types of agents, action agents / plan and execute agents
# Action agents: Take action based on all the data of the prompt up till that point

import os
import pprint # to make it easier to read data
from dotenv import load_dotenv

# load in the env variables
load_dotenv()

# get the env variable relevant and needed
os.getenv('OPENAI_API_KEY')

#import get_all_tool_names module from langchain agents
# these are all the tools we have access to for our agents
from langchain.agents import get_all_tool_names, load_tools, initialize_agent
from langchain.llms import OpenAI

# setup openAI
llm = OpenAI(temperature=0) # makes sure the data is consistent and not creative as we read data

# use the pprint library and setup
pp = pprint.PrettyPrinter(indent=2)
# get all the 
pp.pprint(get_all_tool_names())

# prompt we are trying to get up to date information from
prompt = "What was the year Nelson Mandela was born? what is that year raised to the power of 3?"

# setup the tools we will be using to get the answer
# NOTE: Some tools use an LLM so we need to pass the LLM along with the tools to help as a dependecy
# as some have the LLM create code that are executable
tools = load_tools(["llm-math", "wikipedia"], llm=llm) # the answer from get_all_tool_names

# agent setup
# different agent types (7)
# zero-shot-react-description - uses the react framework to know what tool to use based soley on the tools description
# i.e all the tools passed we have have descriptions and this looks at that to decide which one based on the section in the prompt
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run(prompt)