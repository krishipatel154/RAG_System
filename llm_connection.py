from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# load dotenv
load_dotenv()

# load model
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ask question
question = str(input("enter your question: "))
response = llm.invoke([HumanMessage(content=question)])
print(response.content)