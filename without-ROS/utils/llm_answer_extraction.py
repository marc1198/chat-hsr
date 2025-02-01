import pathlib
import json
import re

# Man könnte evtl. die ganze Datei aufteilen in zwei: History Handling und LLM Answer Extraction

class LLMAnswerExtraction:
    def __init__(self):
        self.folder = pathlib.Path(__file__).parent.resolve()

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