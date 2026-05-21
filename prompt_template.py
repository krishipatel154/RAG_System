from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
# ai response -> str
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system","you are a helpful assistant that answers questions in a concise manner."),
    ("human","Explain the following topic: {topic}")
])

# pipeline
chain = prompt | llm | StrOutputParser()
topic = str(input("enter your topic: "))
response = chain.invoke({"topic": topic})

print(response)