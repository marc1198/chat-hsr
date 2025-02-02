import pathlib
import json
import re

# Man könnte evtl. die ganze Datei aufteilen in zwei: History Handling und LLM Answer Extraction

class LLMAnswerExtraction:
    def __init__(self):
        self.folder = pathlib.Path(__file__).parent.resolve()    

    def get_removed_objects(self, feedback, object_remove):
        """Parses LLM response and updates the list of removed objects"""
        try:
            results = self.extract_json(feedback["content"])
            if results:
                
                # Check the structure of the JSON response
                if isinstance(results, dict) and ('object' in results or 'objects' in results) and 'task' in results:
                    # Process and print results
                    object_names = results.get("object", results.get("objects"))
                    if isinstance(object_names, str):
                        object_names = [object_names]
                    for obj in object_names:
                        if obj not in object_remove:
                            object_remove.append(obj)
                    
                    task = results["task"]
                    print(f"\n\n{object_names = }, {task = }\n")
                    
                else:
                    print("Invalid structure in the JSON response. Continuing conversation.")

            return object_remove

        except json.JSONDecodeError:
            print("Error decoding JSON. Please try again.")
            return object_remove
        
        
    def extract_json(self, text):
        """Extract JSON response from LLM output"""  
        json_pattern = r'(\{.*?\})'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_string = match.group(1)
            try:
                return json.loads(json_string)
            except json.JSONDecodeError:
                print("Failed to decode JSON from the response.")
                return None
        return None