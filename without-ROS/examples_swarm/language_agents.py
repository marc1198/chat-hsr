from swarm import Swarm, Agent
import os
from openai import OpenAI

ollama_client=OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
)

client = Swarm(client=ollama_client)

english_agent = Agent(
    name="English Agent",
    instructions="You only speak English.",
    model="qwen2.5:14b"
)

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You only speak Spanish.",
    model="qwen2.5:14b"
)


def transfer_to_spanish_agent():
    """Transfer spanish speaking users immediately."""
    return spanish_agent

english_agent.functions.append(transfer_to_spanish_agent)

def transfer_to_english_agent():
    """Transfer english speaking users immediately."""
    return english_agent

spanish_agent.functions.append(transfer_to_english_agent)


messages = [{"role": "user", "content": "Hola. ¿Como estás?"}]
response = client.run(agent=english_agent, messages=messages)
print(response.messages[-1]["content"])

messages = [{"role": "user", "content": "Sorry i don't speak spanish."}]
response = client.run(agent=spanish_agent, messages=messages)
print(response.messages[-1]["content"])