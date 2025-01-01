import json
import requests
import re
import pathlib


folder = pathlib.Path(__file__).parent.resolve()
# NOTE: ollama must be running for this to work, start the ollama docker container
ollama_ip = "localhost"
model = 'qwen2.5:32b'  # TODO: update this for the model you wish to use
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
def chat(messages):
    stream = True

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
    json_pattern = r'({.*?})'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1)
    return None

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

# Initialize messages with few-shot examples
def initialize_messages(system_message):
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

    # Possible format I might want to use
    """Instructions: {System Instruction} 
    Previous Data: {Memory} 
    Reference Data: {Retrieved Information} 
    Respond in the specified JSON format: {JSON Format with descriptions} 
    Please replicate the examples to generate the answer: 
    {k examples} 
    Given question: '''{Question}''', provide the process leading to the answer: 
    """
        
    
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
                f"Forget all previous lists completely and only use this one. "
                f"Please list the items you believe match my request and check each one against the current list. " # ADDED
                #f"Double-check each object against this list before responding. "
                f"If an object is not available, inform me and suggest any truly similar alternatives as a simple list (not in JSON). "
                f"Make sure every item in your JSON response is from the current list. "
                f"Here is my request: {user_input}"
            )
        })

        print("Sending prompt to LLM")
        feedback = chat(messages)
        messages.append(feedback)

        # Check if the response contains valid JSON object
        try:
            json_string = extract_json(feedback["content"])
            
            if json_string:
                results = json.loads(json_string)

                # Check the structure of the JSON response
                if isinstance(results, dict) and ('object' in results or 'objects' in results) and 'task' in results:
                    

                    # Process and print results
                    object_name = results.get("object", results.get("objects"))
                    object_name = object_name if isinstance(object_name, list) else [object_name]
                    task = results["task"]

                    print(f"\n\n{object_name = }, {task = }\n")

                    # Update object history:
                    for single_object in object_name:
                        update_object_history(single_object, task)
                    
                    # Update object_remove and detected objects
                    for obj in object_name:
                        if obj not in object_remove:
                            object_remove.append(obj)

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


