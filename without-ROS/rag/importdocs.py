import chromadb
from functions import readtextfiles, chunksplitter, chunksplitter_conversation_pairs, getembedding

chromaclient = chromadb.HttpClient(host="chromadb", port=8000)
textdocspath = "./history_documents/evaluation"
text_data = readtextfiles(textdocspath)

collections = chromaclient.list_collections()
for collection_name in collections:
  if collection_name == "buildragwithpython":
    print("Collection already existed")    
    chromaclient.delete_collection("buildragwithpython")

collection = chromaclient.get_or_create_collection(name="buildragwithpython", metadata={"hnsw:space": "cosine"}  )


for filename, text in text_data.items():
  chunks = chunksplitter_conversation_pairs(text)
  embeds = getembedding(chunks)
  chunknumber = list(range(len(chunks)))
  ids = [filename + str(index) for index in chunknumber]
  metadatas = [{"source": filename} for index in chunknumber]
  collection.add(ids=ids, documents=chunks, embeddings=embeds, metadatas=metadatas)


#print(chromaclient.list_collections())
print(f"Document Embeddings added successfully to collection. Content: \n {collection.get()}")
