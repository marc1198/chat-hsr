import pathlib
import json
import re

# Man könnte evtl. die ganze Datei aufteilen in zwei: History Handling und LLM Answer Extraction

class HistoryHandler:
    def __init__(self, relative_history_file="../rag/history_documents/message_history_new.txt"):
        self.folder = pathlib.Path(__file__).parent.resolve()
        self.history_file = f"{self.folder}/{relative_history_file}"

    # Right now not used!! used from main.py. I think later I will do this in agents.py!
    def load_system_message(self, system_message_file):
        with open(system_message_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    def initialize_messages(self, system_message):
        """Initialize messages with few-shot examples"""
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

        # Possible format I might want to use
        """Instructions: {System Instruction} 
        Previous Data: {Memory} 
        Reference Data: {Retrieved Information} 
        Respond in the specified JSON format: {JSON Format with descriptions} 
        Please replicate the examples to generate the answer: 
        {k examples} 
        Given question: '''{Question}''', provide the process leading to the answer: 
        """
        
        return messages
    
    def initialize_messages_swarm(self, system_message):
        """Initialize messages with few-shot examples (without system prompt for usage with swarm)"""
        messages = [            
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

    def save_history(self, messages):
        """Save message history to a file."""
        messages_with_time_steps = self.add_time_steps(messages)  # Add time steps to message structure --> {time_step: "", user: "", assistent: ""}
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(messages_with_time_steps, f, indent=4, ensure_ascii=False)

    def add_time_steps(self, messages):
        """Adds time steps to the message history, ignoring system messages."""
        new_messages = []
        time_step = 1
        user_message = None

        for message in messages:
            if message.get("role") == "system":  # Ignore system messages
                continue
            if message.get("role") == "user":
                user_message = message # Store user message for pairing
            elif message.get("role") == "assistant" and user_message: # Pair with previous user message
                new_messages.append({"time_step": str(time_step), "role": user_message["role"], "content": user_message["content"]})
                new_messages.append({"time_step": str(time_step), "role": message["role"], "content": message["content"]})
                time_step += 1
                user_message = None

        return new_messages