import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    embedding_function = get_embeddings_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    results = db.similarity_search_with_score(query_text, k=5)

    # context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    context_text = "\n\n".join([f"Document {doc.metadata.get('id', 'Unknown')}:\n{doc.page_content}" for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3.2",host="127.0.0.1", port=11435)
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()