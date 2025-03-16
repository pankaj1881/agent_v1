import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
import os,json
import random
import string
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")

from langchain_core.documents import Document

documents = [
    Document(
        page_content="Pankaj Sakharkar was social servant.",
      
    ),
    Document(
        page_content="there is scheme named pankaj sakharkar yojana for economically weaken people,",
    
    ),
    Document(
        page_content="Only thos e who have income below 100000 RS can apply for this scheme.",
   
    ),
    Document(
        page_content="Any one who age above 18 can apply fro this scheme only once in life time.",
     
    ),
    Document(
        page_content="Document required for pankaj sakharkar scheme is adhar card and PAN card.",
    
    ),
]

from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

## VectorStores
from langchain_community.vectorstores import FAISS

vectorstore=FAISS.from_documents(documents,embedding=embeddings)


# File path for the JSON file
json_file_path = r"data.json"

# Function to load data from the JSON file

def load_data():
    if os.path.exists(json_file_path):
        with open(json_file_path, 'r') as file:
            return json.load(file)
    else:
        # Initialize structure if file doesn't exist
        return {
            "my_account": {
                "name": "",
                "amount": 0,
                "account_number": "",
                "user_id": "",
                "password": ""
            },
            "beneficiaries": []
        }

# Function to save data to the JSON file
def save_data(data):
    with open(json_file_path, 'w') as file:
        json.dump(data, file, indent=4)

# Function to get account balance
def get_account_balance(account_number):
    """Function to fetch Account balance.
    Account_number is Mandatory! """
    data = load_data()
    
    # Fetch balance by account_number
    for beneficiary in data["beneficiaries"]:
        if beneficiary.get("account_number") == account_number:
            return f"Account number {account_number}, Name {beneficiary['beneficiary_name']}  balance is {beneficiary.get('amount')} rs"
    
    return "Account number not found."

# Function to transfer money and update balance
def transfer_money(beneficiary_name=None, account_number=None, amount=0):
    """
    Function to transfer amount to beneficiary.
    """
    data = load_data()
    
    # Get sender's account details
    sender_balance = data["my_account"]["amount"]
    
    if sender_balance < amount:
        return "Insufficient balance for this transfer."
    
    # Find the beneficiary
    for beneficiary in data["beneficiaries"]:
        if (
            (account_number and beneficiary.get("account_number") == account_number) or
            (beneficiary_name and beneficiary.get("beneficiary_name") == beneficiary_name)
        ):
            # ✅ If beneficiary name is missing, take it from JSON data
            if not beneficiary_name:
                beneficiary_name = beneficiary.get("beneficiary_name")

            # Update balances
            beneficiary["amount"] += amount
            data["my_account"]["amount"] -= amount
            
            # Save updated data
            save_data(data)
            
            return (f"{amount} rs successfully transferred to "
                    f"{beneficiary_name} ({account_number}). "
                    f"Your new balance is {data['my_account']['amount']} rs.")
    
    return "Beneficiary not found."


# Function to add credits or debits and update balance
def update_balance(account_number, amount):
    """Function to credit or debit from an account"""
    data = load_data()

    # Check if account number exists
    for beneficiary in data["beneficiaries"]:
        if beneficiary["account_number"] == account_number:
            new_balance = beneficiary["amount"] + amount  # Adding or subtracting the amount
            beneficiary["amount"] = new_balance
            save_data(data)
            return f"Account {account_number} updated. New balance is {new_balance} rs."
    
    return "Account number not found."

# Function to add new beneficiary
def add_beneficiary(account_number, beneficiary_name, amount = 0 ):
    """Function to add new beneficiary to JSON"""
    data = load_data()
    
    # Check if beneficiary already exists
    for beneficiary in data["beneficiaries"]:
        if beneficiary["account_number"] == account_number:
            return "Beneficiary with this number already exists."

    # Add new beneficiary to the list
    new_beneficiary = {
        "beneficiary_name": beneficiary_name,
        "amount": amount,
        "account_number": account_number   # Create an account number based on beneficiary number
    }
    
    data["beneficiaries"].append(new_beneficiary)
    save_data(data)
    
    return f"{beneficiary_name} successfully added."

def generate_thread_id():
    return ''.join(random.choices(string.ascii_letters + string.digits, k=10))


# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


from langchain.tools import tool
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

class RagToolSchema(BaseModel):
    question: str

@tool(args_schema=RagToolSchema)
def retriever_tool(question):
    """Tool to Retrieve Semantically Similar documents to answer User Questions,"""
    print("INSIDE RETRIEVER NODE")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    ## RAG


    message = """
    Answer this question using the provided context only.

    {question}

    Context:
    {context}
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    rag_chain={"context":retriever,"question":RunnablePassthrough()}|prompt|llm

    response=rag_chain.invoke(question)

    return response.content

llm_with_tools=llm.bind_tools([retriever_tool,get_account_balance,transfer_money,add_beneficiary])


# Build graph
builder = StateGraph(MessagesState)
builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([retriever_tool,get_account_balance,transfer_money, add_beneficiary]))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges(
    "tool_calling_llm",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", END)

from langgraph.checkpoint.memory import MemorySaver

memory=MemorySaver()

graph=builder.compile(checkpointer=memory)


# Define a function that uses your model
def query_model(prompt: str, config: dict):
    messages = [HumanMessage(content=prompt)]
    # Assuming `graph.invoke()` works as expected in your environment
    messages = graph.invoke({"messages": messages}, config=config)
    return messages['messages']

# Initialize session state for login status
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Load user credentials from JSON
data = load_data()
stored_user_id = data["my_account"]["user_id"]
stored_password = data["my_account"]["password"]

# ✅ Sidebar for Login/Logout
with st.sidebar:
    st.header("User Authentication")

    if not st.session_state.logged_in:
        user_id = st.text_input("User ID")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if user_id == stored_user_id and password == stored_password:
                st.session_state.logged_in = True
                st.success("Login successful!")
               
            else:
                st.error("Invalid User ID or Password")

    else:
        st.write(f"Logged in as **{data['my_account']['name']}**")

        # ✅ Logout button
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.success("Logged out successfully!")
            

# ✅ If logged in → Show chatbot
if st.session_state.logged_in:
    # Chatbot UI components
    st.title("BANK CHATBOT")
    st.write("Enter your query below to interact with the model.")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    if 'config' not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": generate_thread_id()}}

    if st.button("Refresh memory!!"):
        # Generate a new random thread_id
        st.session_state.config = {"configurable": {"thread_id": generate_thread_id()}}
        st.success(f"Session ID: {st.session_state.config['configurable']['thread_id']}")

    # User input for the prompt
    user_input = st.text_input("Enter your message!")

    # # Default configuration
    config = {"configurable": {"thread_id": "1"}}

    if st.button("Submit"):
        # Call the model and get the response
        with st.spinner("Getting response..."):
            messages = query_model(user_input, config)
        
        # Append the human message to the conversation history
        for m in messages:
            if isinstance(m, HumanMessage):
                st.session_state.conversation_history.append({"role": "Human", "content": m.content})
            elif isinstance(m, AIMessage):
                st.session_state.conversation_history.append({"role": "AI", "content": m.content})
            elif isinstance(m, ToolMessage):
                st.session_state.conversation_history.append({"role": "Tool", "content": m.content})
        
        # Display the conversation history
        st.subheader("Conversation History:")
        for message in st.session_state.conversation_history:
            if message["role"] == "Human":
                st.markdown(f"**Human**: {message['content']}")
            elif message["role"] == "AI":
                st.markdown(f"**AI**: {message['content']}")
            elif message["role"] == "Tool":
                st.markdown(f"**Tool**: {message['content']}")
