# Langchain library with LLMs (OpenAI)

import os
from dotenv import load_dotenv

# load in the env variables
load_dotenv()

# get the env variable relevant and needed
os.getenv('OPENAI_API_KEY')

#import openai module from langchain
from langchain.llms import OpenAI

# setup an instance of the open ai class
# Tempreture: determines randomness and creativity (between 0-1) - where closer to zero is deterministic and 1 is random creative 
# Model: you can specify but if you dont, when initializing the class, by default it uses: text-davinci-003
llm = OpenAI(temperature=1)

#prompt
prompt = "What would be a good company name be for a company that makes colorful socks?"

# to chain we can use the generate function from langchain
# print(llm(prompt))

# calling with the same prompt 5 times i.e [prompt, prompt, prompt, prompt, prompt] - sequencial calls?
# think of it as 5 calls and then having a list available with the data of each
result = llm.generate([prompt]*10) 
# loop over the results to get the data
for name in result.generations:
    print(name[0].text)