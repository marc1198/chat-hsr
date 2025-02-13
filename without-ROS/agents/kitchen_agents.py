from swarm import Swarm, Agent
from openai import OpenAI

import sys
import pathlib


this_folder = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(this_folder.parent))

from utils.history_handler import HistoryHandler
from utils.object_detection import ObjectDetection

history_handler = HistoryHandler()
object_detector = ObjectDetection()

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
swarm_ollama_client = Swarm(client=ollama_client)

def send_email(recipient, subject, body):
    print("Sending email...")
    print(f"To: {recipient}")
    print(f"Subject: {subject}")
    print(f"Body: {body}")
    return "Sent!"

def retrieve_history(context_variables, searched_object): #full_history muss von context_variables kommen
    """Find out where the object is now based on the history and the searched object"""
    # TODO:
    # RAG einbinden:
    # importdocs(context_variables["messages")
    # result = search(searched_object) # evtl. besser wenn ich die ganze user message reingebe? --> messages[-1; "user"] oder so ähnlich?
    # return result
    print("Aus history function. Verfügbare Context Variables: ")
    print(context_variables["messages"])
    return_string = f"{searched_object} was not found in the history. But I also didn't get a chat history :("
    print(return_string)
    return return_string

def get_current_objects(**kwargs):
    """Important: Just write the query in the arguments ("kwargs")"""

    # Edit: Die Funktion sollte eigentlich automatisch aufgerufen werden, bzw. sollte die nicht im Agent sein, sondern in der main.py und als context_variable übergeben werden!!
    objects = object_detector.detect_objects([]) # Edit: Eigentlich braucht er die object_remove)
    #messages.append({"role": "user", "content": f'The objects available are: {objects}.'})
    return objects

routing_agent = Agent(
    name="Routing Agent",
    instructions="Determine which agent is best suited to handle the user's request, and transfer the conversation to that agent.",
    model="llama3.2:3b"
)


task_planning_agent = Agent(
    name="Task Planning Agent",
    instuctions = "Check if you find the object in the list. If not, try to find truly similar objects to the user that he could use for the desired context. When finished, return to routing_agent",
    functions=[get_current_objects],
    model="llama3.2:3b"
)

knowledge_base_agent = Agent(
    name="Knowledge Base Agent",
    instructions="Retrieve the chat history of a given object to find out where the object is now. This agent is helpful when a user asked where a certain object is or where it was put. It is also helpful when a user wants to write an email.",
    functions=[retrieve_history, send_email],
    model="llama3.2:3b"
)

def transfer_back_to_router(**kwargs): # Hier will ich eigentlich, dass der Agent wieder an den Routing Agent antwortet und der dann die Antwort an den Benutzer zusammenfasst. Aber vlt. auch unnötig kompliziert.
    """
    Call this function if a user is asking about a topic that is not handled by the current agent.
    Important: Just write the query in the arguments ("kwargs")
    """
    #"""Call this function everytime at the end of the agents answer. The user requests should always go to the routing_agent first."""
    print("transfered to routing agent")
    return routing_agent

def transfer_to_task_planning(**kwargs):
    """
    Use this function when the user requests an object, or when his requests requires planning or the execution of an action.
    It should be invoked for requests that involve initiating, organizing, or carrying out tasks.
    Important: Just write the query in the arguments ("kwargs")

    """

    print("transfered to Task Planning Agent")
    return task_planning_agent

def transfer_to_knowledge_base(**kwargs):
    """
    Use this function when the user's request seeks to retrieve information about 
    the state, location, or historical context of objects.
    It should be invoked for inquiries that rely on historical or context-based data.
    Important: Just write the query in the arguments ("kwargs")
    """

    print("transfered to Knowledge Base Agent")
    return knowledge_base_agent


routing_agent.functions = [transfer_to_task_planning, transfer_to_knowledge_base]
#task_planning_agent.functions.append(transfer_back_to_router)
#history_agent.functions.append(transfer_back_to_router)

#messages = [{"role": "user", "content": "Can i have banana?"}]
#response = swarm_ollama_client.run(agent=routing_agent, messages=messages)

#print(response.messages[-1]["content"])