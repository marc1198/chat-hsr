import os
import re
import ollama

def readtextfiles(path):
  text_contents = {}
  directory = os.path.join(path)

  for filename in os.listdir(directory):
    if filename.endswith(".txt"):
      file_path = os.path.join(directory, filename)

      with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

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
    #pattern = r'(\{\s*"role":\s*"(?:user|assistant)".*?\})' # Regex for txt file without time_step (only "user": "", "assistant": "")
    pattern = r'(\{\s*"time_step":\s*"\d+"\s*,\s*"role":\s*"(?:user|assistant)".*?\})' # Regex for txt file including time_step
    matches = re.findall(pattern, text, re.DOTALL)
    
    chunks = []
    current_chunk = []

    for match in matches:
        current_chunk.append(match)

        # Falls das aktuelle Chunk eine User- und eine Assistant-Nachricht enthält, speichern wir es
        if len(current_chunk) == 2:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    return chunks
    

def getembedding(chunks):
  embeds = ollama.embed(model="nomic-embed-text", input=chunks)
  return embeds.get('embeddings', [])
