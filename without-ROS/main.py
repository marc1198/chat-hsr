import pathlib
import json

#from swarm.repl import run_demo_loop
from agents.kitchen_agents import routing_agent, swarm_ollama_client
from utils.chat import ChatModel
from utils.history_handler import HistoryHandler
from utils.llm_answer_extraction import LLMAnswerExtraction
from utils.object_detection import ObjectDetection


folder = pathlib.Path(__file__).parent.resolve()

# NOTE: ollama must be running for this to work, start the ollama docker container and pull the LLM used
chat_model = ChatModel()
history_handler = HistoryHandler()
llm_answer_extractor = LLMAnswerExtraction()
object_detector = ObjectDetection()

#Load system message
data= f'{folder}/system_message.txt'
with open(data, 'r', encoding='utf-8') as data_message:
    system_message = data_message.read()
    

def run_chat_loop():
    #Initialize messages for ongoing dialogue and list of objects to remove from list   
    object_remove = []
    messages = history_handler.initialize_messages_swarm()

    # Initial object detection (EDIT: Right now turned off, because agent should do it themselves)
    print(f'object remove: {object_remove}')
    #objects = object_detector.detect_objects(object_remove)
    #messages.append({"role": "user", "content": f'The objects available are: {objects}.'})
   
    while True:
        user_input = input("Enter your prompt: ")
        if not user_input:
            exit()
        
        print()
        messages = [{
            "role": "user",
            "content": user_input        
        }]
        """messages.append({
            "role": "user", 
            "content": (
                #f"This is the new list of detected objects, it overwrites all previous lists of objects: {objects}. "
                #f"Forget all previous lists completely and only use this one. "
                #f"Please list the items you believe match my request and check each one against the current list. " # ADDED
                ##f"Double-check each object against this list before responding. "
                #f"If an object is not available, inform me and suggest any truly similar alternatives as a simple list (not in JSON). "
                #f"Make sure every item in your JSON response is from the current list. "
                f"Here is my request: {user_input}"
            )
        })"""
        print("Sending prompt to LLM")
        
        # New message sent over swarm_ollama_client
        feedback_swarm = swarm_ollama_client.run(agent=routing_agent, messages=messages)

        # LLM Answer received
        #feedback_ollama = chat_model.chat(messages) # Original message sent over ollama
        feedback = {"role": "assistant", "content": feedback_swarm.messages[-1]["content"]}

        messages.append(feedback)
        history_handler.save_history(messages)

        # Update removed objects      
        object_remove = llm_answer_extractor.get_removed_objects(feedback, object_remove)
        objects = object_detector.detect_objects(object_remove)
        print(f'object remove: {object_remove}')
        print(f'objects: {objects} \n')

def process_and_print_streaming_response(response):
    content = ""
    last_sender = ""

    for chunk in response:
        if "sender" in chunk:
            last_sender = chunk["sender"]

        if "content" in chunk and chunk["content"] is not None:
            if not content and last_sender:
                print(f"\033[94m{last_sender}:\033[0m", end=" ", flush=True)
                last_sender = ""
            print(chunk["content"], end="", flush=True)
            content += chunk["content"]

        if "tool_calls" in chunk and chunk["tool_calls"] is not None:
            for tool_call in chunk["tool_calls"]:
                f = tool_call["function"]
                name = f["name"]
                if not name:
                    continue
                print(f"\033[94m{last_sender}: \033[95m{name}\033[0m()")

        if "delim" in chunk and chunk["delim"] == "end" and content:
            print()  # End of response message
            content = ""

        if "response" in chunk:
            return chunk["response"]


def pretty_print_messages(messages) -> None:
    for message in messages:
        if message["role"] != "assistant":
            continue

        # print agent name in blue
        print(f"\033[94m{message['sender']}\033[0m:", end=" ")

        # print response, if any
        if message["content"]:
            print(message["content"])

        # print tool calls in purple, if any
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call["function"]
            name, args = f["name"], f["arguments"]
            arg_str = json.dumps(json.loads(args)).replace(":", "=")
            print(f"\033[95m{name}\033[0m({arg_str[1:-1]})")


def run_demo_loop(
    starting_agent, context_variables=None, stream=False, debug=False
) -> None:
    client = swarm_ollama_client
    print("Starting Swarm CLI 🐝")

    messages = []
    if context_variables is None:
        context_variables = {"messages": messages} 
    agent = starting_agent

    while True:
        user_input = input("\033[90mUser\033[0m: ")
        messages.append({"role": "user", "content": user_input})
        recent_messages=messages[-6:] # Edit: Hier noch sicherstellen, dass nur die wichtigen Messages drin sind (sowas wie relevant_messages=filter_messages() --> filtert tool_calls und handoffs raus aus messages); vlt. sogar für history agent nur spannend die nachrichten von task_planning_agent
        context_variables["messages"]=messages

        response = client.run(
            agent=agent,
            messages=recent_messages,
            context_variables=context_variables or {},
            stream=stream,
            debug=debug,
        )

        if stream:
            response = process_and_print_streaming_response(response)
        else:
            pretty_print_messages(response.messages)

        messages.extend(response.messages)
        #agent = response.agent # We always want the routing agent to be the first one receiving the message

if __name__ == "__main__":
    #Start-Sequence
    print("Hi, I am here to assist you with the objects on the table. Please tell me what you need.")
    run_demo_loop(routing_agent, stream=False, debug=False)
    #run_chat_loop()



