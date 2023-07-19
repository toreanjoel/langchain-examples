# Custom models with Hugging face and Langchain

import os
from dotenv import load_dotenv

# load in the env variables
load_dotenv()

# get the env variable relevant and needed
os.getenv('HUGGINGFACEHUB_API_TOKEN')

#import HuggingFaceHub module from langchain
from langchain import HuggingFaceHub

# setup an instance with the model we want as the repo-id instantiated with it
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",
    model_kwargs={
            "temperature": 0, # deterministic or creative as all models
            "max_length": 64 # the limit of the max length of text
        }
    )

prompt = "What are good fitness tips?"

print(llm(prompt))