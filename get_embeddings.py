from langchain_huggingface import HuggingFaceEmbeddings
def get_embeddings_function():
    # Specify the model name you want to use for embeddings, such as `sentence-transformers/all-MiniLM-L6-v2`
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings