import sys, chromadb, ollama
import requests, json
from functions import getembedding


ollama_ip = "localhost"
model = 'llama3.1'  # TODO: update this for the model you wish to use

#API chat completion call
def chat(messages, stream = True):
    r = requests.post(
        f"http://{ollama_ip}:11434/api/chat",
        json={"model": model, "messages": messages, "stream": stream, "temperature": 0.0, "options": {"num_ctx": 4096}
        },
        stream=True,
    )
    r.raise_for_status()
    
    if stream:
        output = ""

        for line in r.iter_lines():
            body = json.loads(line)
            if "error" in body:
                raise Exception(body["error"])
            if body.get("done") is False:
                message = body.get("message", "")
                content = message.get("content", "")
                output += content
                #the response streams one token at a time, print that as we receive it
                print(content, end="", flush=True)

            if body.get("done", False):
                message["content"] = output
                print()
                return message # Return the answer from the LLM
    else:
        body = r.json()
        if "error" in body:
            raise Exception(body["error"])

        message = body.get("message", {})
        content = message.get("content", "")
        print(content)
        return {"content": message.get("content", "")}


# 1. Chroma Setup
chromaclient = chromadb.HttpClient(host="chromadb", port=8000)
collection = chromaclient.get_or_create_collection(name="buildragwithpython")

# 2. Embed query
query = " ".join(sys.argv[1:])
queryembed=getembedding(query)

# 3. Get related documents from query embedding
query_results = collection.query(query_embeddings=queryembed, n_results=3)
if query_results['documents']:  # Überprüfen, ob Ergebnisse vorhanden sind
    relateddocs = '\n\n'.join(query_results['documents'][0])
else:
    print("Keine passenden Dokumente gefunden.")

print(f"related Docs: {relateddocs}")

system_prompt = "You are given a message history of interactions between a user and an robot assistant: The robot assistant has the ability to see objects on a table and either hand them over to the user or place them at specific locations. Your task is to analyze this conversation history and provide the user with information about it. Key details to consider: 1. Time Order: - The provided question-answer pairs are NOT in chronological order but instead sorted by relevancy to the user’s query. - Each message contains a \"time_step\" field, where a higher time_step value means the event happened later. - When reconstructing events, always use the highest time_step for an object to determine its last known state. 2. robot's Response Format: - The robot assistant always responds with a JSON string containing the task it performed (e.g., \"handover\" or \"placement\") and, if applicable, the final location of the object. - If the format of this JSON is valid, assume that the last recorded location of an object in the assistant's response is where it currently is. Example: If the robot placed cutlery in the sink at time_step: 6 but later placed it in the bin at time_step: 7, then the cutlery’s last known location is the bin. Please ensure your response is clear, concise, easy to understand for a human and does not contain JSON."
prompt = f"{query} - Answer that question using the following message history as your resource: {relateddocs}." # {system_prompt}

messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]

#noragoutput = ollama.generate(model="qwen2.5:32b", prompt=query, stream=False)
#print(f"Answered without RAG: {noragoutput['response']}")
print("---")
ragoutput = chat(messages, stream=False)
#ragoutput = ollama.generate(model="qwen2.5:32b", prompt=prompt, stream=False)
#print(f"Answered with RAG: {ragoutput['content']}")