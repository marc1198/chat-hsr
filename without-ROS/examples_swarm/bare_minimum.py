from swarm import Swarm, Agent
from openai import OpenAI

ollama_client=OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
)


client = Swarm(client=ollama_client)

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
    model="qwen2.5:14b"
)

messages = [{"role": "user", "content": "Hi!"}]
response = client.run(agent=agent, messages=messages)

print(response.messages[-1]["content"])


