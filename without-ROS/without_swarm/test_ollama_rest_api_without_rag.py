import json
import requests
import re
import pathlib


folder = pathlib.Path(__file__).parent.resolve()
# NOTE: ollama must be running for this to work, start the ollama docker container and pull the LLM used
ollama_ip = "localhost"
model = 'llama3.1:latest'  # TODO: update this for the model you wish to use
# Best: qwen2.5:32b & gemma2:27b
# Old but good: llama3.1:latest

print(model)

#Load system message
data= f'{folder}/system_message.txt'
with open(data, 'r', encoding='utf-8') as data_message:
    system_message = data_message.read()

ollama_ip = "localhost"

# Dictionary to save history of all objects that were handled by the robot
object_history = {}
tools = [
  {
    "type": "function",
    "function": {
      "name": "get_object_history",  # Replace with the actual function name
      "description": "Retrieves the task history of a given object to find out where the object is now.",
      "parameters": {
        "type": "object",
        "properties": {
          "object_name": {
            "type": "string",
            "description": "The name of the object to retrieve history for."
          }
        },
        "required": ["object_name"]
      }
    }
  }
]

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

### LLM Tool Functions ### 
def get_object_history(object_name: str) -> list:
  """
  Retrieves the task history of a given object to find out where the object is now.

  Args:
    object_name: The name of the object to retrieve history for.

  Returns:
    list: A list containing the history of the object. 
          Returns ["No history available."] if no task was applied to this object before.
  """
  for key in object_history:
    if object_name in key:
      return object_history[key]
  return ["No history available."]


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
    refered_messages = """[
        {
            "role": "user",
            "content": "I want a power drill."
        },
        {
            "role": "assistant",
            "content": "There is a power drill available in the current list of objects. I will hand it over to you. {\"object\": \"035_power_drill\", \"task\": \"handover\"}."
        },
        {
            "role": "user",
            "content": "I need some balls on the shelf."
        },
        {
            "role": "assistant",
            "content": "There are different balls available in the current list of objects: mini soccer ball, softball, baseball, tennis ball, racquetball, golf ball. What balls would you like?"
        },
        {
            "role": "user",
            "content": "tennis ball and golf ball"
        },
        {
            "role": "assistant",
            "content": "There is a tennis ball and a golf ball available in the current list of objects. I will place them on the shelf for you. {\"objects\": [\"056_tennis_ball\", \"058_golf_ball\"], \"task\": \"placement\"}."
        },
        {
            "role": "user",
            "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_table', '002_plate', '003_cutlery', '004_glass_mug', '005_spatula', '006_pan', '007_candy_wrapper', '008_salt_shaker', '009_pepper_shaker']. Here is my request: I just finished dinner, please clear the dining table. "
        },
        {
            "role": "assistant",
            "content": "Based on your request to clear the dining table, I will handle the following objects:\n\n- Place the plate in the Sink.\n- Place the cutlery in the Sink.\n- Place the glass mug in the Sink.\n- Place the spatula in the Sink.\n- Place the candy wrapper in the Trash Can.\n\nHere is the JSON output for each task:\n\n{\n  \"objects\": [\"002_plate\"],\n  \"task\": \"placement\",\n  \"location\": \"Sink\"\n}\n\n{\n  \"objects\": [\"003_cutlery\"],\n  \"task\": \"placement\",\n  \"location\": \"Sink\"\n}\n\n{\n  \"objects\": [\"004_glass_mug\"],\n  \"task\": \"placement\",\n  \"location\": \"Sink\"\n}\n\n{\n  \"objects\": [\"005_spatula\"],\n  \"task\": \"placement\",\n  \"location\": \"Sink\"\n}\n\n{\n  \"objects\": [\"007_candy_wrapper\"],\n  \"task\": \"placement\",\n  \"location\": \"Trash Can\"\n}"
        },
        {
            "role": "user",
            "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_table', '006_pan', '008_salt_shaker', '009_pepper_shaker']. Here is my request: please also put the pan in the sink and salt, pepper to the foot shelf"
        },
        {
            "role": "assistant",
            "content": "Based on your request, here are the actions I will take:\n\n- Place the pan in the Sink.\n- Place the salt shaker on the Food Shelf.\n- Place the pepper shaker on the Food Shelf.\n\nHere is the JSON output for each task:\n\n{\n  \"objects\": [\"006_pan\"],\n  \"task\": \"placement\",\n  \"location\": \"Sink\"\n}\n\n{\n  \"objects\": [\"008_salt_shaker\", \"009_pepper_shaker\"],\n  \"task\": \"placement\",\n  \"location\": \"Food Shelf\"\n}"
        },
        {
            "role": "user",
            "content": "I want a power drill."
        },
        {
            "role": "assistant",
            "content": "There is a power drill available in the current list of objects. I will hand it over to you. {\"object\": \"035_power_drill\", \"task\": \"handover\"}."
        },
        {
            "role": "user",
            "content": "I need some balls on the shelf."
        },
        {
            "role": "assistant",
            "content": "There are different balls available in the current list of objects: mini soccer ball, softball, baseball, tennis ball, racquetball, golf ball. What balls would you like?"
        },
        {
            "role": "user",
            "content": "tennis ball and golf ball"
        },
        {
            "role": "assistant",
            "content": "There is a tennis ball and a golf ball available in the current list of objects. I will place them on the shelf for you. {\"objects\": [\"056_tennis_ball\", \"058_golf_ball\"], \"task\": \"placement\"}."
        },
        {
            "role": "user",
            "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_table', '002_clothes', '003_brush', '004_candy_wrapper', '005_pen', '006_packaging', '007_scissors', '008_salt']. Here is my request: Please hand me the brush and tidy up the rest of the living room. "
        },
        {
            "role": "assistant",
            "content": "Sure, I will hand you the brush and take care of tidying up the rest of the living room.\n\nHere's the JSON for handing over the brush:\n```json\n{\n  \"objects\": [\"003_brush\"],\n  \"task\": \"handover\"\n}\n```\n\nAnd here are the actions to tidy up:\n\n- Place the candy wrapper in the Trash Can.\n- Place the pen in the Storage Box.\n- Place the packaging in the Trash Can.\n- Place the scissors in the Storage Box.\n\nHere is the JSON for placing the objects:\n```json\n{\n  \"objects\": [\"004_candy_wrapper\", \"006_packaging\"],\n  \"task\": \"placement\",\n  \"location\": \"Trash Can\"\n}\n{\n  \"objects\": [\"005_pen\", \"007_scissors\"],\n  \"task\": \"placement\",\n  \"location\": \"Storage Box\"\n}\n```"
        },
        {
            "role": "user",
            "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_table', '002_clothes', '008_salt']. Here is my request: please put the clothes in the storage box and the salt in the food shelf"
        },
        {
            "role": "assistant",
            "content": "Sure, I will place the clothes in the Storage Box and the salt in the Food Shelf.\n\nHere is the JSON for placing the objects:\n\n```json\n{\n  \"objects\": [\"002_clothes\"],\n  \"task\": \"placement\",\n  \"location\": \"Food Shelf\"\n}\n{\n  \"objects\": [\"008_salt\"],\n  \"task\": \"placement\",\n  \"location\": \"Food Shelf\"\n}\n```"
        },
            {
            "role": "user",
            "content": "I want a power drill."
        },
        {
            "role": "assistant",
            "content": "There is a power drill available in the current list of objects. I will hand it over to you. {\"object\": \"035_power_drill\", \"task\": \"handover\"}."
        },
        {
            "role": "user",
            "content": "I need some balls on the shelf."
        },
        {
            "role": "assistant",
            "content": "There are different balls available in the current list of objects: mini soccer ball, softball, baseball, tennis ball, racquetball, golf ball. What balls would you like?"
        },
        {
            "role": "user",
            "content": "tennis ball and golf ball"
        },
        {
            "role": "assistant",
            "content": "There is a tennis ball and a golf ball available in the current list of objects. I will place them on the shelf for you. {\"objects\": [\"056_tennis_ball\", \"058_golf_ball\"], \"task\": \"placement\"}."
        },
        {
            "role": "user",
            "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_plant', '002_table', '003_plate', '004_fruit', '005_laptop_computer', '006_plastic_wrapper', '007_mug', '008_mouse', '009_computer', '010_candy_wrapper', '011_packaging_booklet', '012_pepper']. Here is my request: Please clear my desk, leaving only the essentials for work. "
        },
        {
            "role": "assistant",
            "content": "Based on your request to clear the desk and leave only the essentials for work, I will assume you want to keep items like the laptop computer (`005_laptop_computer`), mouse (`008_mouse`), and any documents or booklets that might be essential (`011_packaging_booklet`). The following objects can be removed:\n\n- 001_plant\n- 002_table (assuming this is not the desk itself)\n- 003_plate\n- 004_fruit\n- 006_plastic_wrapper\n- 007_mug\n- 009_computer (if you only need the laptop and mouse, we'll remove this one)\n- 010_candy_wrapper\n- 012_pepper\n\nPlease confirm if these items should be removed:\n\n- Plant (`001_plant`)\n- Table (`002_table`)\n- Plate (`003_plate`)\n- Fruit (`004_fruit`)\n- Plastic Wrapper (`006_plastic_wrapper`)\n- Mug (`007_mug`)\n- Computer (`009_computer`)\n- Candy Wrapper (`010_candy_wrapper`)\n- Pepper (`012_pepper`)"
        },
        {
            "role": "user",
            "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_plant', '002_table', '003_plate', '004_fruit', '005_laptop_computer', '006_plastic_wrapper', '007_mug', '008_mouse', '009_computer', '010_candy_wrapper', '011_packaging_booklet', '012_pepper']. Here is my request: Please clear my desk, leaving only the essentials for work. "
        },
        {
            "role": "assistant",
            "content": "Based on your request to clear the desk and leave only the essentials for work, I will assume you want to keep items like the laptop computer (`005_laptop_computer`), mouse (`008_mouse`), and any documents or booklets that might be essential (`011_packaging_booklet`). The following objects can be removed:\n\n- 001_plant\n- 002_table (assuming this is not the desk itself)\n- 003_plate\n- 004_fruit\n- 006_plastic_wrapper\n- 007_mug\n- 009_computer (if you only need the laptop and mouse, we'll remove this one)\n- 010_candy_wrapper\n- 012_pepper\n\nHere is the JSON for placing these items:\n\n```json\n{\n  \"objects\": [\"006_plastic_wrapper\", \"010_candy_wrapper\"],\n  \"task\": \"placement\",\n  \"location\": \"Trash Can\"\n}\n{\n  \"objects\": [\"003_plate\", \"007_mug\"],\n  \"task\": \"placement\",\n  \"location\": \"Sink\"\n}\n{\n  \"objects\": [\"001_plant\", \"002_table\", \"004_fruit\", \"009_computer\", \"012_pepper\"],\n  \"task\": \"placement\",\n  \"location\": \"Storage Box\"\n}\n```\n\nPlease confirm if this is correct."
        },
            {
            "role": "user",
            "content": "I want a power drill."
        },
        {
            "role": "assistant",
            "content": "There is a power drill available in the current list of objects. I will hand it over to you. {\"object\": \"035_power_drill\", \"task\": \"handover\"}."
        },
        {
            "role": "user",
            "content": "I need some balls on the shelf."
        },
        {
            "role": "assistant",
            "content": "There are different balls available in the current list of objects: mini soccer ball, softball, baseball, tennis ball, racquetball, golf ball. What balls would you like?"
        },
        {
            "role": "user",
            "content": "tennis ball and golf ball"
        },
        {
            "role": "assistant",
            "content": "There is a tennis ball and a golf ball available in the current list of objects. I will place them on the shelf for you. {\"objects\": [\"056_tennis_ball\", \"058_golf_ball\"], \"task\": \"placement\"}."
        },
        {
            "role": "user",
            "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_plant', '002_table', '003_plate', '004_fruit', '005_laptop_computer', '006_plastic_wrapper', '007_mug', '008_mouse', '009_computer', '010_candy_wrapper', '011_packaging_booklet', '012_pepper']. Here is my request: Please clear my desk, leaving only the essentials for work. "
        },
        {
            "role": "assistant",
            "content": "Based on your request to clear the desk and leave only the essentials for work, I will move the following items:\n\n- **Trash Can:** plastic wrapper (006_plastic_wrapper), candy wrapper (010_candy_wrapper), packaging booklet (011_packaging_booklet).\n- **Sink:** plate (003_plate), mug (007_mug).\n\nThe essentials for work that will remain on the desk are: laptop computer (005_laptop_computer), mouse (008_mouse), and computer (009_computer). \n\nHere is the JSON output for the task:\n\n{\n  \"objects\": [\"006_plastic_wrapper\", \"010_candy_wrapper\", \"011_packaging_booklet\"],\n  \"task\": \"placement\",\n  \"location\": \"Trash Can\"\n}\n\n{\n  \"objects\": [\"003_plate\", \"007_mug\"],\n  \"task\": \"placement\",\n  \"location\": \"Sink\"\n}"
        },
        {
            "role": "user",
            "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_plant', '002_table', '004_fruit', '005_laptop_computer', '008_mouse', '009_computer', '012_pepper']. Here is my request: this looks really good, there's only two more things missing which should not be on the desk: please put the fruit and the pepper in the ffood shelf"
        },
        {
            "role": "assistant",
            "content": "Sure, I will place the fruit and the pepper on the food shelf.\n\nHere is the JSON output for the task:\n\n{\n  \"objects\": [\"004_fruit\", \"012_pepper\"],\n  \"task\": \"placement\",\n  \"location\": \"Food Shelf\"\n}"
        },
        {
            "role": "user",
            "content": "This is the new list of detected objects, it overwrites all previous lists of objects: ['001_table', '002_plate', '003_cutlery', '004_glass_mug', '005_spatula', '006_pan', '007_candy_wrapper', '008_salt_shaker', '009_pepper_shaker']. Here is my request: please put the pepper shaker where it belongs"
        },
    
        {
            "role": "assistant",
            "content": "Based on the current list of objects, I will place the pepper shaker back in its proper location. Since its specific home isn't listed among the possible locations, I'll assume a general storage area for now.\n\n{\n  \"objects\": [\"009_pepper_shaker\"],\n  \"task\": \"placement\",\n  \"location\": \"Storage Box\"\n}"
        },
    ]"""
    messages = [
        {"role": "system", "content": system_message},
        
        # Few-shot example 1
        {"role": "user", "content": "I want a power drill."},
        {"role": "assistant", "content": "There is a power drill available in the current list of objects. I will hand it over to you. {\"object\": \"035_power_drill\", \"task\": \"handover\"}."},
        
        # Few-shot example 2
        {"role": "user", "content": "I need some balls on the shelf."},
        {"role": "assistant", "content": "There are different balls available in the current list of objects: mini soccer ball, softball, baseball, tennis ball, racquetball, golf ball. What balls would you like?"},
        {"role": "user", "content": "tennis ball and golf ball"},
        {"role": "assistant", "content": "There is a tennis ball and a golf ball available in the current list of objects. I will place them on the shelf for you. {\"objects\": [\"056_tennis_ball\", \"058_golf_ball\"], \"task\": \"placement\"}."}
    ]
    return messages, refered_messages

def main():
    #Start-Sequence
    print("Hi, I am here to assist you with the objects on the table. Please tell me what you need.")
    
    #Initialize messages for ongoing dialogue and list of objects to remove from list   
    object_remove = []
    messages, refered_messages = initialize_messages(system_message)

    # Initial object detection
    print(f'object remove: {object_remove}')
    objects = object_detection(object_remove)
    #Edit: messages.append({"role": "user", "content": f'The objects available are: {objects}.'})
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
                #Edit: f"This is the new list of detected objects, it overwrites all previous lists of objects: {objects}. "
                #f"Forget all previous lists completely and only use this one. "
                ##f"Please list the items you believe match my request and check each one against the current list. " # ADDED
                #f"Double-check each object against this list before responding. "
                #f"If an object is not available, inform me and suggest any truly similar alternatives as a simple list (not in JSON). "
                ##f"Make sure every item in your JSON response is from the current list. "
                f"Here is a conversation history of a chat with you and a user: {refered_messages}. Please check for the answer to the following request in this conversation history. Here is my request: {user_input}" #Edit
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
                #Edit:messages.append({"role": "user", "content": f'This is the new list of detected objects, it overwrites all previous lists of objects: {objects}.'})

                print("\n")
            else:
                print("Invalid structure in the JSON response. Continuing conversation.")

        except json.JSONDecodeError:
            print("Error decoding JSON. Please try again.")

if __name__ == "__main__":
    main()


