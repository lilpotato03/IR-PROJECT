import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embeddings import get_embeddings_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful research assistant. Answer the following question based only on the context provided. If the answer is not explicitly in the context, provide an informed response based on your general knowledge.

Context:
{context}

Question:
{question}

Provide a detailed answer, summarizing key points from the context and ensuring all relevant information is included.
"""

def query_rag(query_text: str):
    embedding_function = get_embeddings_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n".join([f"Document {doc.metadata.get('id', 'Unknown')}:\n{doc.page_content}" for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.2", host="127.0.0.1", port=11435)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    return response_text, sources

# Streamlit app
st.title("Research Assistant Chatbot")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history in a chat-like format
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Input field at the bottom of the page
user_query = st.text_input("Enter your question:", key="user_input")
submit_button = st.button("Send")

if submit_button and user_query:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.write(user_query)

    with st.spinner("Fetching response..."):
        response, sources = query_rag(user_query)

    # Display the full response at once
    response_placeholder = st.chat_message("assistant")
    response_placeholder.write(response)  # Display the full response at once

    # Add the response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display sources
    st.markdown("**Sources:**")
    st.write(", ".join(source for source in sources if source))

    # Clear input after submission
    # st.session_state["user_input"] = ""

elif submit_button:
    st.warning("Please enter a question.")
