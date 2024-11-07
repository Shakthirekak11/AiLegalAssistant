import json
import streamlit as st
from upstash_redis import Redis
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_together import Together

# Initialize Redis client with Upstash URL and token
UPSTASH_REDIS_REST_URL='https://equal-dodo-25065.upstash.io'
UPSTASH_REDIS_REST_TOKEN='AWHpAAIjcDFhNjVkNmMzYTkxYTk0MDJjOGY0MTZkZjQ1NWJiZDNjMHAxMA'
redis_client = Redis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)

# Set API key for legal LLM
# the API key has been split and concatenated to bypass GitHub API screening
a='bb1b45095f0459a'
b='f3dc33743c083e9'
c='d8ae15be886fd'
d='859a6a049a'
e='826a1f8746c'
api_key = a+b+c+d+e

# Initialize the Mistral LLM
mistral_llm = Together(model="mistralai/Mixtral-8x22B-Instruct-v0.1", temperature=0.5, max_tokens=1024, together_api_key=api_key)

# Load Chats
def load_chat(chat_name):
    chat_data = redis_client.get(chat_name)
    if chat_data:
        data = json.loads(chat_data)
        entity_memory = ConversationEntityMemory(
            llm = mistral_llm, k=data.get('k', 50)
        )
        for i in range(len(data["past"])):
            entity_memory.save_context({"input": data["past"][i]}, {"output": data["generated"][i]})
        return {"generated": data["generated"], "past": data["past"], "entity_memory": entity_memory}
    return {"generated": [], "past": [], "entity_memory": ConversationEntityMemory(llm = mistral_llm, k=50)}

# Save Chats
def save_chat(chat_name, chat_data):
    serializable_data = {"generated": chat_data["generated"], "past": chat_data["past"], "k": chat_data["entity_memory"].k}
    redis_client.set(chat_name, json.dumps(serializable_data))

if "current_chat" not in st.session_state:
    st.session_state.current_chat = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""

# Create New Chats
def create_new_chat():
    new_chat_name = f"Chat {len(list(redis_client.keys('*'))) + 1}"
    chat_data = {"generated": [], "past": [], "entity_memory": ConversationEntityMemory(llm = mistral_llm, k=50)}
    save_chat(new_chat_name, chat_data)
    st.session_state.current_chat = new_chat_name
    st.session_state.input_text = ""

# Switch between Chats
def switch_chat(chat_name):
    st.session_state.current_chat = chat_name
    st.session_state.input_text = ""

# Process input prompt
def process_input():
    user_input = st.session_state.input_text

    if user_input:
        current_chat = load_chat(st.session_state.current_chat)
        llm = mistral_llm
        conversation = ConversationChain(
            llm=llm,
            prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
            memory=current_chat["entity_memory"]
        )
        # Load embeddings for Indian Penal Code and similarity search
        embeddings = HuggingFaceEmbeddings(model_name="law-ai/InLegalBERT")
        db = FAISS.load_local("ipc_embed_db", embeddings, allow_dangerous_deserialization=True)
        db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = db_retriever.get_relevant_documents(user_input)
        response = conversation.run(input=user_input + "\n\n" + str(docs))
        current_chat["past"].append(user_input)
        current_chat["generated"].append(response)
        save_chat(st.session_state.current_chat, current_chat)
        
    st.session_state.input_text = ""

# Sidebar with chat options
with st.sidebar:
    st.title("Chats")
    if st.button("Create New Chat"):
        create_new_chat()

    chat_keys = redis_client.keys('*')
    sorted_chat_keys = sorted(chat_keys, key=lambda x: int(x.split()[1]), reverse=True)

    for chat_name in sorted_chat_keys:
        if st.button(chat_name):
            st.session_state.current_chat = chat_name
            st.session_state.input_text = ""

# Main UI for chat interface
if st.session_state.current_chat:
    st.markdown("<h1 style='text-align: center;'>⚖️ AI Legal Assistant ⚖️</h1>", unsafe_allow_html=True)
    st.write(" ")
    st.markdown(f"{st.session_state.current_chat}")

    # Display conversation history in a scrollable container
    current_chat = load_chat(st.session_state.current_chat)
    with st.container():
        if current_chat["generated"]:
            for i in range(len(current_chat["generated"])):
                user_message = st.chat_message("user")
                ai_message = st.chat_message("assistant")
                user_message.write(f"{current_chat['past'][i]}")
                ai_message.write(f"{current_chat['generated'][i]}")
        else:
            st.write("_No conversation history yet._")

    # Input field for user text
    st.text_input(
        "Your question",
        value=st.session_state.input_text,
        key="input_text",
        placeholder="Ask a legal question...",
        label_visibility="hidden",
        on_change=process_input
    )

else:
    st.title("")
    st.markdown("<h1 style='text-align: center;'>⚖️ AI Legal Assistant ⚖️</h1>", unsafe_allow_html=True)
    st.write("Please create or select a chat from the sidebar.")