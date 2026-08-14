from dotenv import load_dotenv
load_dotenv()

from langchain_core import __version__ as core_version    
from importlib.metadata import version
from langchain_groq import ChatGroq

lg_version = version("langgraph")

def main():
    print("Hello from learnrag!")
    print(f"langchain-core version: {core_version}")
    print(f"langgraph version: {lg_version}")
    llm=ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response=llm.invoke("Write a poem about the ocean.")
    print(response) 

if __name__ == "__main__":
    main()