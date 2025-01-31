import chromadb
import torch
from sentence_transformers import SentenceTransformer

# 1. Load the Sentence Transformer model and move it to the GPU:
#model = SentenceTransformer('all-MiniLM-L6-v2')
#model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)


# Funktionieren nicht plug&play mit dem Code (aber haben sehr gute Performance!)
#model = SentenceTransformer('nvidia/NV-Embed-v2', trust_remote_code=True)
#model = SentenceTransformer("BAAI/bge-multilingual-gemma2", model_kwargs={"torch_dtype": torch.float16})
#model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1",  trust_remote_code=True)



if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)
    print("GPU is available, model moved to GPU")
else:
    print("GPU is not available, using CPU")
    device = torch.device("cpu") # explicitly set device to cpu

# 2. Define a custom embedding function (Corrected for Chroma 0.4.16+):
class SentenceTransformerEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):  # Correct signature: self, input
        embeddings = self.model.encode(input)
        return embeddings.tolist()
    
embedding_function = SentenceTransformerEmbeddingFunction(model)

# 3. Initialize Chroma with the custom embedding function:
#chroma_client = chromadb.Client() # local client (also works with the embeddings)
chroma_client = chromadb.HttpClient(host="chromadb", port=8000) # HTTP client via chromadb Docker 

for collection_name in chroma_client.list_collections():
  if collection_name == "my_collection":
    print("Collection already existed")    
    chroma_client.delete_collection("my_collection")

collection = chroma_client.create_collection(name="my_collection", embedding_function=embedding_function) # Pass the embed_function


# 4. Prepare documents - Embed them first and then convert embeddings to NumPy before adding:
documents = [
    "pineapple",
    "oranges",
    "second world war",
    "trousers"
]
ids = ["id1", "id2", "id3", "id4"]

embeddings = embedding_function(documents)

# 5. Add documents (now using the GPU for embeddings):
collection.add(
    documents=documents,
    ids=ids,
    embeddings=embeddings # Add the numpy array
)

#print(collection.get())


# 6. Query (Corrected - embed query using the custom function and convert to numpy):
query_texts=["pants"]
query_embeddings = embedding_function(query_texts)

results = collection.query(
    query_embeddings=query_embeddings, 
    n_results=3
)

print(results)