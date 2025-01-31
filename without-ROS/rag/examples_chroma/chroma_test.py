import chromadb
import torch
from sentence_transformers import SentenceTransformer

# 1. Load the Sentence Transformer model and move it to the GPU:
model = SentenceTransformer('all-MiniLM-L6-v2')
if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print("GPU is available, model moved to GPU")
else:
    print("GPU is not available, using CPU")
    device = torch.device("cpu")

# 2. Define a custom embedding function:
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        embeddings = self.model.encode(input)
        return embeddings.tolist()
    
embedding_function = SentenceTransformerEmbeddingFunction(model)

# 3. Initialize Chroma with the custom embedding function:
chroma_client = chromadb.Client() # local client

for collection_name in chroma_client.list_collections():
  if collection_name == "my_collection":
    print("Collection already existed")    
    chroma_client.delete_collection("my_collection")

collection = chroma_client.create_collection(name="my_collection", embedding_function=embedding_function) # Pass the embed_function


# 4. Add documents (now using the GPU for embeddings):
documents = [
    "pineapple",
    "oranges",
    "second world war",
]
ids = ["id1", "id2", "id3"]

collection.add(
    documents=documents,
    ids=ids,
)

#print(collection.get())

# 5. Query
query_texts=["war"]

results = collection.query(
    query_texts=query_texts,
    n_results=3
)

print(results)
