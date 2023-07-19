# Storing and Retrieving Chat history
import os
import pprint # to make it easier to read data
from dotenv import load_dotenv

# load in the env variables
load_dotenv()

# get the env variable relevant and needed
os.getenv('OPENAI_API_KEY')

# these are all the tools we have access to for our agents
from langchain import OpenAI, ConversationChain
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict

# setup a class that has functions to initialize messages that we can setup
history = ChatMessageHistory()
history.add_user_message("hello, lets talk about football")
history.add_ai_message("Okay, lets do it")

# use the to dict function to take all the accumulated appended messages
# we can store this for later if we wish to save
dict = messages_to_dict(history.messages)
print(dict)

# We want to load in the messags that has a history initially that we want to load
# as if we loaded it from a DB or file
new_messages = messages_from_dict(dict)

# setup the class for openAI
llm = OpenAI(temperature=0)
# initialize the chat history first so we have the data
history = ChatMessageHistory(messages=new_messages)
# setup the buffer for storing the messages
buffer = ConversationBufferMemory(chat_memory=history)
# add the conversation data
conversation = ConversationChain(llm=llm, memory=buffer, verbose=True)
# start a new input with no context that will use the memory for context
print(conversation.predict(input="Do you like the sport?"))