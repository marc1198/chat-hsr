import json
import requests
import re
import pathlib


folder = pathlib.Path(__file__).parent.resolve()
# NOTE: ollama must be running for this to work, start the ollama docker container and pull the LLM used
ollama_ip = "localhost"
model = 'llama3.1'  # TODO: update this for the model you wish to use
# Best: qwen2.5:32b & gemma2:27b
# Old but good: llama3.1:latest

print(model)

#Load system message
data= f'{folder}/system_message.txt'
with open(data, 'r', encoding='utf-8') as data_message:
    system_message = data_message.read()


# Dictionary to save history of all objects that were handled by the robot
object_history = {}

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



# Object detection of available objects on the table
def object_detection(object_remove):
    objects_file = f'{folder}/objects_available.txt'
    with open(objects_file, 'r', encoding='utf-8') as all_objects:
        objects = all_objects.read().replace("\n", "")
        objects = [obj.strip(' " ') for obj in objects.split(',')]
        objects = [obj for obj in objects if obj not in object_remove]
    print("Object detection done")
    return objects

# Extract JSON response from LLM output
def extract_json(text):
    json_pattern = r'(\{.*?\})'
    matches = re.findall(json_pattern, text, re.DOTALL)
    extracted_jsons = []

    for match in matches:
        try:
            extracted_jsons.append(json.loads(match))
        except json.JSONDecodeError:
            print("Failed to decode JSON from response.")
    
    return extracted_jsons if extracted_jsons else None

def update_object_history(object_name, task):
  if object_name in object_history:
    object_history[object_name].append(task)
  else:
    object_history[object_name] = [task] 


def save_message_history(messages, file_path=f"{folder}/rag/history_documents/message_history_new.txt"):
    """Save message history to a file."""
    messages_with_time_steps = add_time_steps(messages)  # Add time steps to message structure --> {time_step: "", user: "", assistent: ""}
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(messages_with_time_steps, f, indent=4, ensure_ascii=False)


def add_time_steps(messages):
    """Adds time steps to the message history, ignoring system messages."""
    new_messages = []
    time_step = 1
    user_message = None  # Initialize to None

    for message in messages:
        if message.get("role") == "system":  # Ignore system messages
            continue

        if message.get("role") == "user":
            user_message = message  # Store user message for pairing
        elif message.get("role") == "assistant" and user_message:  # Pair with previous user message
            new_messages.append({
                "time_step": str(time_step),
                "role": user_message["role"],
                "content": user_message["content"]
            })
            new_messages.append({
                "time_step": str(time_step),
                "role": message["role"],
                "content": message["content"]
            })
            time_step += 1
            user_message = None # Reset user_message after pairing
        elif message.get("role") == "assistant" and not user_message: #Handle the case where an assistant message comes before a user one.
            #Logically, this should not happen, but if it does, it will be ignored to keep the time_step logic consistent.
            pass #Do nothing

    return new_messages

# Initialize messages with few-shot examples
def initialize_messages(system_message):
    messages = [
        {"role": "system", "content": system_message},
        
        # Few-shot example 1 (low-level)
        {"role": "user", "content": "I want a power drill."},
        {"role": "assistant", "content": "There is a power drill available in the current list of objects. I will hand it over to you. {\"object\": \"035_power_drill\", \"task\": \"handover\"}."},
        
        # Few-shot example 2 (low-level)
        {"role": "user", "content": "I need some balls on the shelf."},
        {"role": "assistant", "content": "There are different balls available in the current list of objects: mini soccer ball, softball, baseball, tennis ball, racquetball, golf ball. What balls would you like?"},
        {"role": "user", "content": "tennis ball and golf ball"},
        {"role": "assistant", "content": "There is a tennis ball and a golf ball available in the current list of objects. I will place them on the shelf for you. {\"objects\": [\"056_tennis_ball\", \"058_golf_ball\"], \"task\": \"placement\"}."},

        # Few-shot example 3 (high-level)
        {"role": "user", "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_table', '002_picture_frame', '003_orange_peel', '004_chair', '005_mug', '006_toothpick', '007_apple', '008_charger', '009_milk', '010_bread', '011_backpack', '012_flowers']. Here is my request: Please tidy up the workspace, but leave any personal decorative items."},
        {"role": "assistant", "content": "Based on your request, I will remove the relevant objects from the workspace while leaving personal decorative items, such as the picture frame and flowers, untouched. Here's how I'll categorize and handle the objects:\n\n- Place the mug in the Sink.\n- Place the orange peel and the toothpick in the Trash Can.\n- Place the apple and bread on the Food Shelf.\n- Place the milk in the Fridge.\n- Place the charger and backpack in the Storage Box.\n- No action for the table, picture frame, chair, and flowers.\n\nHere is the JSON output for each task:\n\n{\n  \"objects\": [\"005_mug\"],\n  \"task\": \"placement\",\n  \"location\": \"Sink\"\n}\n\n{\n  \"objects\": [\"003_orange_peel\", \"006_toothpick\"],\n  \"task\": \"placement\",\n  \"location\": \"Trash Can\"\n}\n\n{\n  \"objects\": [\"007_apple\", \"010_bread\"],\n  \"task\": \"placement\",\n  \"location\": \"Food Shelf\"\n}\n\n{\n  \"objects\": [\"009_milk\"],\n  \"task\": \"placement\",\n  \"location\": \"Fridge\"\n}\n\n{\n  \"objects\": [\"008_charger\", \"011_backpack\"],\n  \"task\": \"placement\",\n  \"location\": \"Storage Box\"\n}"}
    ]

    return messages

def main():
    #Start-Sequence
    print("Hi, I am here to assist you with the objects on the table. Please tell me what you need.")
    
    #Initialize messages for ongoing dialogue and list of objects to remove from list   
    object_remove = []
    messages = initialize_messages(system_message)

    # Initial object detection
    print(f'object remove: {object_remove}')
    objects = object_detection(object_remove)
    messages.append({"role": "user", "content": f'The objects available are: {objects}.'})
    # old_object_message_id = len(messages)-1
    
    #new object-detection after chain-of-thought
    objects = object_detection(object_remove)
    #messages.append({"role": "user", "content": f'This is the new list of detected objects, it overwrites all previous lists of objects: {objects}.'})

          
    while True:
        user_input = input("Enter your prompt: ")
        if not user_input:
            exit()
        
        print()
        messages.append({
            "role": "user", 
            "content": (
                f"This is the new list of detected objects, it overwrites all previous lists of objects: {objects}. "
                #f"Forget all previous lists completely and only use this one. "
                ##f"Please list the items you believe match my request and check each one against the current list. " # ADDED
                #f"Double-check each object against this list before responding. "
                #f"If an object is not available, inform me and suggest any truly similar alternatives as a simple list (not in JSON). "
                ##f"Make sure every item in your JSON response is from the current list. "
                f"Here is my request: {user_input}"
            )
        })

        print("Sending prompt to LLM")
        feedback = chat(messages)
        messages.append(feedback)

        # Save message history to file
        save_message_history(messages)


        # Check if the response contains valid JSON object
        try:
            json_blocks = extract_json(feedback["content"])
            
            if json_blocks:
                for results in json_blocks:
                    # Check the structure of the JSON response
                    if isinstance(results, dict) and ('object' in results or 'objects' in results) and 'task' in results:
                        

                        # Process and print results
                        object_name = results.get("object", results.get("objects"))
                        object_name = object_name if isinstance(object_name, list) else [object_name]
                        task = results["task"]
                        if 'location' in results:
                            location = results["location"]
                            print(f"\n\n{object_name = }, {task = }, {location = }")
                        else:
                            print(f"\n\n{object_name = }, {task = }")
                        
                        # Update object_remove and detected objects
                        for obj in object_name:
                            if obj not in object_remove:
                                object_remove.append(obj)
                
                print("\n\n")
                objects = object_detection(object_remove)
                print(f'object remove: {object_remove}')
                print(f'objects: {objects}')
                messages.append({"role": "user", "content": f'This is the new list of detected objects, it overwrites all previous lists of objects: {objects}.'})

                print("\n")
            else:
                print("Invalid structure in the JSON response. Continuing conversation.")

        except json.JSONDecodeError:
            print("Error decoding JSON. Please try again.")

if __name__ == "__main__":
    main()


