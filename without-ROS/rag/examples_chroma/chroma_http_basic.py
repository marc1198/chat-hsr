import chromadb

chromaclient = chromadb.HttpClient(host="chromadb", port=8000)

collections = chromaclient.list_collections()
for collection_name in collections:
  if collection_name == "buildragwithpython":
    print("Collection already existed")    
    chromaclient.delete_collection("buildragwithpython")

collection = chromaclient.get_or_create_collection(name="buildragwithpython", metadata={"hnsw:space": "cosine"})
model = collection.get_model()
print(model)

collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

print(collection.get())
