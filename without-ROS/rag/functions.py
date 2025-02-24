import os
import re
import ollama
import json
from sentence_transformers import SentenceTransformer
import torch 
import ast


def readtextfiles(path):
  text_contents = {}
  directory = os.path.join(path)
  file_directory = os.path.dirname(os.path.abspath(__file__))
  os.chdir(file_directory)

  for filename in os.listdir(directory):
    if filename.endswith(".txt"):
      file_path = os.path.join(directory, filename)

      with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

      if(content) != "":
        text_contents[filename] = content

  return text_contents

def chunksplitter(text, chunk_size=100):
  words = re.findall(r'\S+', text)

  chunks = []
  current_chunk = []
  word_count = 0

  for word in words:
    current_chunk.append(word)
    word_count += 1

    if word_count >= chunk_size:
      chunks.append(' '.join(current_chunk))
      current_chunk = []
      word_count = 0

  if current_chunk:
    chunks.append(' '.join(current_chunk))

  return chunks

def chunksplitter_conversation_pairs(text):
  data = json.loads(text) # Parse JSON
  chunks = []
  current_chunk = []

  for entry in data:
    if entry["role"] == "user":
      # Start a new chunk with user input
      current_chunk = [entry]
    elif entry["role"] == "assistant" and current_chunk:
      # Pair the assistant response with the user input
      current_chunk.append(entry)
      chunks.append(json.dumps(current_chunk, indent=2)) # Store as JSON string
      current_chunk = [] # Reset for next pair

  return chunks

def chunksplitter_conversation_pairs_without_brackets(text):
    data = json.loads(text)  # Parse JSON
    chunks_brackets = []
    chunks_no_brackets = []
    current_chunk = []

    for entry in data:
        if entry["role"] == "user":
            # Start a new chunk with user input
            current_chunk = [entry]
        elif entry["role"] == "assistant" and current_chunk:
            # Pair the assistant response with the user input
            current_chunk.append(entry)

            # Create chunk with brackets (like original function)
            chunks_brackets.append(json.dumps(current_chunk, indent=2))

            # Create chunk without brackets (like _without_brackets function)
            chunk_string_no_bracket = ""
            for item in current_chunk:
                chunk_string_no_bracket += json.dumps(item, indent=2) + " "
            chunks_no_brackets.append(chunk_string_no_bracket.strip())

            current_chunk = []  # Reset for next pair

    return chunks_brackets, chunks_no_brackets


"""def chunksplitter_conversation_pairs(text):
    data = json.loads(text)  # Parse JSON
    chunks = []
    current_chunk = []

    for entry in data:
        if entry["role"] == "user":
            # Start a new chunk with user input
            current_chunk = [entry]
        elif entry["role"] == "assistant" and current_chunk:
            # Pair the assistant response with the user input
            current_chunk.append(entry)
            chunks.append(json.dumps(current_chunk, indent=2))  # Store as JSON string
            current_chunk = []  # Reset for next pair

    return chunks"""


def getembedding(input):

  # Using Ollama Embedding (nomic)
  #embed = ollama.embed(model="mxbai-embed-large:latest", input=input)['embeddings']
  # 'nomic-embed-text' - alternative model (but probably worse)

  # Using Sentence Transformers
  embed_model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True)
  embed = embed_model.encode(input)

  # Other available SoTa Sentence Transformers Models: 
  # 'BAAI/bge-m3' - precise and fast
  # 'Alibaba-NLP/gte-Qwen2-1.5B-instruct' - precise but slow
  # 'Snowflake/snowflake-arctic-embed-l-v2.0' - alternative model (but probably worse)


  return embed
