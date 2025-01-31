from swarm import Swarm, Agent
from openai import OpenAI

# Was noch fehlt:
    # 1. Test mit context_variables durchführen (ist wichtig zum History oder Objekt Liste weitergeben)
    # 2. Verstehen wie ich Parameter in Funktion/Tool übergebe
        # Geht das eh einfach so plug & play? --> Ja!
        # Hilft es trotzdem den DocString zu erweitern um Params?
    # 3. History aufzeichnen (s. ollama_rest_api bzw. andere Beispiele von mir)
    # 4. Möglichkeit, dass ich chat history an history_agent weitergebe 
        # über context_variables?
        # über dynamische Variable, die in Funktion die History aufruft?
        # oder vlt. sogar direkt in der Message des Routing Agents mit drin?
    # 5. Möglichkeit anstatt direktem Transfer eine neue Transfer-Message des Routing Agents anzugeben
    # 6. RAG einbauen (aus Beispiel ollama RAG) 
        # Vergleich ohne RAG & mit RAG

ollama_client=OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
)
client = Swarm(client=ollama_client)

def retrieve_history(searched_object): #full_history muss von context_variables kommen oder von dynamischer Variable als Funktion??
    """Find out where the object is now based on the history and the searched object"""
    return_string = f"{searched_object} was not found in the history. But I also didn't get a chat history :("
    print(return_string)
    return return_string

triage_agent = Agent(
    name="Triage Agent",
    instructions="Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.",
    model="qwen2.5:14b"
)

current_list_agent = Agent(
    name="Current List Agent",
    instructions="Check if you find the object in the list. If not, try to find truly similar objects to the user that he could use for the desired context. When finished, return to triage_agent",
    model="qwen2.5:14b"
)

history_agent = Agent(
    name="History Agent",
    instructions="Retrieve the chat history of a given object to find out where the object is now. This agent is helpful when a user asked where a certain object is or where it was put.",
    functions=[retrieve_history],
    model="qwen2.5:14b"
)

def transfer_back_to_triage(): # Hier will ich eigentlich, dass der Agent wieder an den Routing Agent antwortet und der dann die Antwort an den Benutzer zusammenfasst. Aber vlt. auch unnötig kompliziert.
    """Call this function if a user is asking about a topic that is not handled by the current agent."""
    #"""Call this function everytime at the end of the agents answer. The user requests should always go to the triage_agent first."""
    print("transfered to triage_agent")
    return triage_agent

def transfer_to_current():
    print("transfered to current_list_agent")
    return current_list_agent

def transfer_to_history():
    print("transfered to history_agent")
    return history_agent


triage_agent.functions = [transfer_to_current, transfer_to_history]
current_list_agent.functions.append(transfer_back_to_triage)
history_agent.functions.append(transfer_back_to_triage)

messages = [{"role": "user", "content": "Where is the apple from before?"}]
response = client.run(agent=triage_agent, messages=messages)

print(response.messages[-1]["content"])