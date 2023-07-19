# Memory and chat bots
import os
from dotenv import load_dotenv

# load in the env variables
load_dotenv()

# get the env variable relevant and needed
os.getenv('OPENAI_API_KEY')

# these are all the tools we have access to for our agents
from langchain import OpenAI, ConversationChain

# The llm setup with the predictive tem[]
llm = OpenAI(temperature=0)

# Setup the conversational chain using the mode to keep memory
conversation = ConversationChain(llm=llm, verbose=True)

# Start a conversation with the initial input of the human to start the convo
print(conversation.predict(input="Hello"))
print(conversation.predict(input="How are you doing today?"))
# print(conversation.predict(input="How are you doing today?")) # keep adding more to chain together more responses

# using terminal inputs
print("Start typing a message to your interactive chat bot...")

# loop and keep track of 3 inputs
# wait for user input
# pass input to in put for the AI and respond with AI result
# go back to human input till iterations are met
for _ in range(0,3):
    human_input = input("You: ")
    ai_response = conversation.predict(input=human_input)
    print(f"AI: {ai_response}")