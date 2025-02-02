import json
import re
import pathlib

from utils.chat import ChatModel
from utils.history_handler import HistoryHandler
from utils.llm_answer_extraction import LLMAnswerExtraction
from utils.object_detection import ObjectDetection

folder = pathlib.Path(__file__).parent.resolve()

# NOTE: ollama must be running for this to work, start the ollama docker container
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
    messages = history_handler.initialize_messages(system_message)

    # Initial object detection
    print(f'object remove: {object_remove}')
    objects = object_detector.detect_objects(object_remove)
    messages.append({"role": "user", "content": f'The objects available are: {objects}.'})
   
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
        
        # LLM Answer received
        feedback = chat_model.chat(messages)
        messages.append(feedback)
        history_handler.save_history(messages)

        # Update removed objects      
        object_remove = llm_answer_extractor.get_removed_objects(feedback, object_remove)
        objects = object_detector.detect_objects(object_remove)
        print(f'object remove: {object_remove}')
        print(f'objects: {objects} \n')


if __name__ == "__main__":
    #Start-Sequence
    print("Hi, I am here to assist you with the objects on the table. Please tell me what you need.")
    run_chat_loop()


