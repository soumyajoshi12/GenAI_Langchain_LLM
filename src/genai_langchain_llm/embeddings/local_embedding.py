from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "Hello this is Soumya joshi",
    "Hello your name is genius"
]

vector = embeddings.embed_documents(texts)

print(vector)