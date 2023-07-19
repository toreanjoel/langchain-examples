# Plan and execute agent
# WHEN TO USE: Long running tasks, long term objective and focus, thinking ahead and execute 1 by 1 maintaining long term objectives
# Creates a plan initially then figures out what it needs to do
# creates an executor that will use the tools to try and get the answer

import os
from dotenv import load_dotenv

# load in the env variables
load_dotenv()

# get the env variable relevant and needed
os.getenv('OPENAI_API_KEY')

#import get_all_tool_names module from langchain agents
# these are all the tools we have access to for our agents
from langchain.agents import Tool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain import SerpAPIWrapper, LLMMathChain, WikipediaAPIWrapper

# setup openAI
llm = OpenAI(temperature=0) # makes sure the data is consistent and not creative as we read data

# Functions for the tools we will use
# We pull in the wrapper functions we can have the run functions used as tools
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
search = SerpAPIWrapper()
wikipedia = WikipediaAPIWrapper()

# Tools
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Useful for when you want to answer questions about current events, also for statistics as a fallback"
    ),
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for when you want to answer questions about facts and statistics"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Useful for when you want to answer questions about math"
    )
]

# prompt we are trying to get up to date information from
# location (country) - search
# Get data of the population of the country - facts and stats
# Work out the answer using math llm math chain
prompt = "What is the last astroid to have flown by earth? How big or wide was the prev astroid to have flown by? Add the size of the two asteroids together in km"

# model
# We use to use the basic model but there is no memory, davici
# with the planner we need memory so we can keep track of the prev info to conitnue improve
model = ChatOpenAI(temperature=0)

# Planner agent that will plan the stepts to get to the answer
planner = load_chat_planner(model)
# Execute agent goes through the steps, executing using the tools it has to get to the best answer
executor = load_agent_executor(model, tools, verbose=True)

# Agent
# create instance that will use the planner and exector
agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)
agent.run(prompt)