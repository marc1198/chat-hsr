#API chat completion call
def chat(messages):
    stream = False

    r = requests.post(
        f"http://{ollama_ip}:11434/api/chat",
        json={"model": model, "messages": messages, "stream": stream, "temperature": 0.0},
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
                return message # Return the answer from the LLM
    else:
        body = r.json()
        if "error" in body:
            raise Exception(body["error"])

        message = body.get("message", {})
        content = message.get("content", "")
        print(content, end="")
        return {"content": message.get("content", "")}