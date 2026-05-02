import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# Load API Key
load_dotenv()

# Manually pulling the key from the environment
api_key = os.getenv("GROQ_API_KEY")

# Initialize the Groq Model
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    groq_api_key=api_key
)

# Define your messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Who is Nelson Mandela?")
]

# Invoke
response = llm.invoke(messages)
print(response.content)