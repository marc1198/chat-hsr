#! /usr/bin/env python3
import os
import openai
import json
import re
import pathlib

#for action_client
#from __future__ import print_function
import difflib
import rospy
import sys

# Brings in the SimpleActionClient
import actionlib
import actionlib_tutorials.msg
from robot_llm.msg import RobotLLMAction, RobotLLMGoal


folder = pathlib.Path(__file__).parent.resolve()
model = 'gpt-3.5-turbo-0125' #Use "gpt-3.5-turbo-0125", "gpt-4o" or other available models if you prefer
print(model)

#Load system message
data= f'{folder}/system_message.txt'
with open(data, 'r', encoding='utf-8') as data_message:
    system_message = data_message.read()

# Access the API_KEY environment variable
openai.api_key = os.getenv('OPEN_API_KEY')
if not openai.api_key:
    raise ValueError("API key not found. Please set the API_KEY environment variable.")

#print(f"The API key is: {openai.api_key}")

#Request to LLM
def get_chat_completion(prompt):
    response = openai.chat.completions.create(
        model=model,  # Use "gpt-3.5-turbo" or other available models if you prefer
        messages=prompt,
        temperature=0.0,
        stream=True,
    )
    #Streaming
    collected_messages = []
    for chunk in response:
        chunk_message = chunk.choices[0].delta.content
        if chunk_message is not None:
            collected_messages.append(chunk_message)
            #the response streams one token at a time, print that as we receive it
            print(chunk_message, end='', flush=True)
    full_reply_content = ''.join(collected_messages)

    return full_reply_content #Return the answer from the LLM

#Object detection of available objects on the table
def object_detection(client):
    rospy.loginfo("Object Detection")
    goal = RobotLLMGoal()
    goal.object_name = ""
    goal.task = "detection"
    client.send_goal(goal)
    rospy.loginfo("Waiting for Object Detection result")
    client.wait_for_result()
    res = client.get_result()  # A FibonacciResult
    res = json.loads(res.result)
    rospy.loginfo("Object detection done")
    return res

#Check for json style response
def extract_json(text):
    json_pattern = r'({.*?})'
    match = re.search(json_pattern, text)
    if match:
        return match.group(1)
    return None

def fibonacci_client():
    client = actionlib.SimpleActionClient('/robot_llm', RobotLLMAction)

    rospy.loginfo("Waiting for ActionServer")
    client.wait_for_server()
    rospy.loginfo("Connected to ActionServer")

    #Start-Sequence
    print("Hi, I am here to assist you with the objects on the table. Please tell me what you need.")
    
    #Initialize messages for ongoing dialogue
    messages = []

    messages.append({"role": "system", "content": system_message}) #Pass system_message to LLM

    #object-detection at the beginning
    res = object_detection(client)    
    objects = res['class_names']
    print(f'objects: {objects}') 
    status = res['status']    # success oder detection_fail
    messages.append({"role": "user", "content": f'The objects available are: {objects}.'})
    
    #Chain-Of-Thought Prompting (2-shot)
    messages.append({"role": "user", "content": 'I want a power drill.'})
    messages.append({"role": "assistant", "content": 'There is a power drill available in the current list of objects. I will hand it over to you. {"object": "035_power_drill", "task": "handover"}.'})
    messages.append({"role": "user", "content": 'I need some balls on the shelf.'})
    messages.append({"role": "assistant", "content": 'There are different balls available in the current list of objects: mini soccer ball, softball, baseball, tennis ball, racquetball, golf ball. What balls would you like?'})
    messages.append({"role": "user", "content": 'tennis ball and golf ball'})
    messages.append({"role": "assistant", "content": 'There is a tennis ball and a golf ball available in the current list of objects. I will place them on the shelf for you. {"objects": ["056_tennis_ball", "058_golf_ball"], "task": "placement"}.'})

    #new object-detection after chain-of-thought
    messages.append({"role": "user", "content": f'I performed a new object detection. You find here the new list of available objects. The new lists overwrites all previous lists of objects: {objects}.'})

    conversation=True
    while not rospy.is_shutdown():
        while conversation:
            user_input = input("Enter your prompt: ")
            if not user_input:
                exit()   
            print()
            
            messages.append({"role": "user", "content": user_input})
            print("Sending prompt to LLM")
            feedback = get_chat_completion(messages)
            messages.append({"role": "assistant", "content": feedback})

            # Check if the response contains valid JSON object
            try:
                #check if response includes json style
                json_string = extract_json(feedback)

                if json_string:
                    json_response = json.loads(json_string)
                    #check final answer
                    if isinstance(json_response, dict) and ('object' in json_response  or 'objects' in json_response) and 'task' in json_response:
                        conversation=False
                else:
                    conversation=True
            except json.JSONDecodeError as e:
                #print("Response is not a valid JSON object.")
                conversation=True

            print("\n")

        rospy.loginfo("Processing LLM result")
        results=json_string
        # Parse the JSON string into a dictionary
        results = json.loads(results)
        # Access the values
        object_name = results.get("object", results.get("objects"))
        object_name = object_name if isinstance(object_name, list) else [object_name]
        task = results["task"]
        
        print(f"{object_name = }, {task = }")
        print("\n")

        for obj in object_name:
             
            goal = RobotLLMGoal()
            goal.object_name = obj
            goal.task = task

            # Sends the goal to the action server.
            rospy.loginfo("Sending goal to grasping pipeline")
            client.send_goal(goal)

            # Waits for the server to finish performing the action.
            rospy.loginfo("Waiting for grasping pipeline results")
            client.wait_for_result()

            # Prints out the result of executing the action
            print(client.get_result())  # success oder object not found

        print("\n")
        
        conversation=True

        #object-detection after every hand-over or placement
        res = object_detection(client)  
        objects = res['class_names']
        status = res['status']    # success oder detection_fail
        print(f'objects: {objects}')  
        messages.append({"role": "user", "content": f'I performed a new object detection. You find here the new list of available objects. The new lists overwrites all previous lists of objects: {objects}.'})

if __name__ == "__main__":
    try:
        # Initializes a rospy node so that the SimpleActionClient can
        # publish and subscribe over ROS.
        rospy.init_node('fibonacci_client_py')
        fibonacci_client()
        #print("Result:", ', '.join([str(n) for n in result.sequence]))
    except rospy.ROSInterruptException:
        print("program interrupted before completion", file=sys.stderr)
