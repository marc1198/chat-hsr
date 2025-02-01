import json
import requests


class ChatModel:
    # Best models: qwen2.5:32b & gemma2:27b & deepseek?
    # Old but good: llama3.1:latest
    # NOTE: ollama must be running for this to work, start the ollama docker container
    def __init__(self, model="qwen2.5:14b", ollama_ip="localhost"):
        self.model = model
        self.ollama_ip = ollama_ip

    #API chat completion call
    def chat(self, messages, stream = True):
        r = requests.post(
            f"http://{self.ollama_ip}:11434/api/chat",
            json={"model": self.model, "messages": messages, "stream": stream, "temperature": 0.0},
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