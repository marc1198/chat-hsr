import sys, chromadb, ollama
from sentence_transformers import SentenceTransformer
import torch 

# 1. Embedding Model Definition
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

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


# 2. Chroma Setup
chromaclient = chromadb.HttpClient(host="chromadb", port=8000)
collection = chromaclient.get_or_create_collection(name="my_collection", embedding_function=embedding_function)


# 3. Search Query
query = " ".join(sys.argv[1:])
#queryembed = ollama.embed(model="nomic-embed-text", input=query)['embeddings']
queryembed = embedding_function(query)



relateddocs = '\n\n'.join(collection.query(query_embeddings=queryembed, n_results=2)['documents'][0])
prompt = f"{query} - Answer that question using the following text as a resource: {relateddocs}"
noragoutput = ollama.generate(model="qwen2.5:14b", prompt=query, stream=False)
print(f"Answered without RAG: {noragoutput['response']}")
print("---")
ragoutput = ollama.generate(model="qwen2.5:14b", prompt=prompt, stream=False)
print(f"query: {query}")
print(f"prompt: {prompt}")

print(f"Answered with RAG: {ragoutput['response']}")