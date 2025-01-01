import json
import requests
import re
import pathlib


folder = pathlib.Path(__file__).parent.resolve()
# NOTE: ollama must be running for this to work, start the ollama docker container
model = 'llama3:latest'  # TODO: update this for the model you wish to use
print(model)

#Load system message
data= f'{folder}/system_message.txt'
with open(data, 'r', encoding='utf-8') as data_message:
    system_message = data_message.read()

TU_wien = False

if TU_wien:
    ollama_ip = "128.131.86.193" 
else:
    ollama_ip = "localhost"

#API chat completion call
def chat(messages):
    r = requests.post(
        f"http://{ollama_ip}:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True, "temperature": 0.0},
        stream=True,
    )
    r.raise_for_status()
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
            return message #Return the answer from the LLM

#Object detection of available objects on the table
def object_detection(object_remove):
    objects = f'{folder}/objects_available.txt'
    with open(objects, 'r', encoding='utf-8') as all_objects:
        objects = all_objects.read().replace("\n", "")
        objects = [obj.strip(' " ') for obj in objects.split(',')]
        objects = [obj for obj in objects if obj not in object_remove]
    print("Object detection done")
    return objects

#Check for json style response
def extract_json(text):
    json_pattern = r'({.*?})'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1)
    return None
    
def main():
    #Start-Sequence
    print("Hi, I am here to assist you with the objects on the table. Please tell me what you need.")
    
    #Initialize messages for ongoing dialogue and list of objects to remove from list   
    messages = []
    object_remove = []

    messages.append({"role": "system", "content": system_message}) #Pass system_message to LLM

    #object-detection at the beginning
    print(f'object remove: {object_remove}')
    objects = object_detection(object_remove)
    messages.append({"role": "user", "content": f'The objects available are: {objects}.'})
    # old_object_message_id = len(messages)-1

    #Chain-Of-Thought Prompting (2-shot)
    messages.append({"role": "user", "content": 'I want a power drill.'})
    messages.append({"role": "assistant", "content": 'There is a power drill available in the current list of objects. I will hand it over to you. {"object": "035_power_drill", "task": "handover"}.'})
    object_remove.append("035_power_drill")
    messages.append({"role": "user", "content": 'I need some balls on the shelf.'})
    messages.append({"role": "assistant", "content": 'There are different balls available in the current list of objects: mini soccer ball, softball, baseball, tennis ball, racquetball, golf ball. What balls would you like?'})
    messages.append({"role": "user", "content": 'tennis ball and golf ball'})
    messages.append({"role": "assistant", "content": 'There is a tennis ball and a golf ball available in the current list of objects. I will place them on the shelf for you. {"objects": ["056_tennis_ball", "058_golf_ball"], "task": "placement"}.'})
    object_remove.append("056_tennis_ball")
    object_remove.append("058_golf_ball")
    
    #new object-detection after chain-of-thought
    objects = object_detection(object_remove)
    messages.append({"role": "user", "content": f'I performed a new object detection. You find here the new list of available objects. The new lists overwrites all previous lists of objects: {objects}.'})

    conversation=True
    while True:        
        while conversation:
            user_input = input("Enter your prompt: ")
            if not user_input:
                exit()
            elif user_input == "save messages":
                with open("protocols/message_protocol.txt", "w") as file: 
                    file.write(str(messages))
            print()
            
            messages.append({"role": "user", "content": user_input})
            print("Sending prompt to LLM")
            feedback = chat(messages)
            messages.append(feedback)

            # Check if the response contains valid JSON object
            try:
                #check if response includes json style
                json_string = extract_json(feedback["content"])

                if json_string:
                    json_response = json.loads(json_string)
                    #check final answer
                    if isinstance(json_response, dict) and ('object' in json_response or 'objects' in json_response) and 'task' in json_response:
                        conversation=False
                else:
                    conversation=True
            except json.JSONDecodeError as e:
                #print("Response is not a valid JSON object.")
                conversation=True

            print("\n")

        print("Processing LLM result")
        results=json_string
        # Parse the JSON string into a dictionary
        results = json.loads(results)
        # Access the values
        object_name = results.get("object", results.get("objects"))
        object_name = object_name if isinstance(object_name, list) else [object_name]
        task = results["task"]
        
        print(f"{object_name = }, {task = }")
        print("\n")

        #remove handed-over or placed objects from current list
        for obj in object_name:
            if obj not in object_remove:
                object_remove.append(obj)

        conversation=True

        #object-detection after every handover or placement
        print(f'object remove: {object_remove}')
        objects = object_detection(object_remove)
        print(f'objects: {objects}')
        messages.append({"role": "user", "content": f'I performed a new object detection. You find here the new list of available objects. The new lists overwrites all previous lists of objects: {objects}.'})

if __name__ == "__main__":
    main()

