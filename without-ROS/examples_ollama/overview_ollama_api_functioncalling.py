# Simple Function Calling (get weather & what is bigger) 
#   - conversation history --> Check
#   - further retrieval of tool call answer (LLM die Antwort vom Tool nochmal verarbeiten lassen) --> Check
#   - no streaming (weiß auch nicht ob das gut geht... Man müsste quasi nochmal das LLM fragen, aber erstmal irrelevant)

import ollama
import requests

model = 'qwen2.5:32b'

# Initializing an empty list for storing the chat messages and setting up the initial system message
chat_messages = []


def get_current_weather(city):
    base_url = f"https://wttr.in/{city}?format=j1"
    response = requests.get(base_url)
    data = response.json()
    return f"Temp in {city}: {data['current_condition'][0]['temp_C']}"


def what_is_bigger(n, m):
    if n > m:
        return f"{n} is bigger"
    elif m > n:
        return f"{m} is bigger"
    else:
        return "they are the same"

# Defining a function to create new messages with specified roles ('user' or 'assistant')
def create_message(message, role):
  return {
    'role': role,
    'content': message
  }

# Function for asking questions - appending user messages to the chat logs before starting the `chat()` function
def ask(message):
  chat_messages.append(
    create_message(message, 'user')
  )
  print(f'\n\n--{message}--\n\n')
  return chat_with_ollama()

def chat_with_ollama_no_functions(user_question):
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'user', 'content': user_question}
        ]
    )
    return response


def chat_with_ollama():
    response = ollama.chat(
        model=model,
        messages=chat_messages,
        tools=[
            {
                'type': 'function',
                'function': {
                    'name': "get_current_weather",
                    'description': "Get the current weather for a city",
                    'parameters': {
                        'type': "object",
                        'properties': {
                            'city': {
                                'type': "string",
                                "description": "City",
                            },
                        },
                        'required': ['city'],
                    },
                },
            },
            {
                'type': "function",
                'function': {
                    "name": "what_is_bigger",
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'n': {
                                'type': "float",
                            },
                            "m": {
                                'type': "float"
                            },
                        },
                        'required': ['n', 'm'],
                    },
                },
            },
        ],
    )
    # Adding the finalized assistant message to the chat log
    if 'message' in response and 'content' in response['message'] and response['message']['content'] != '':
        chat_messages.append(create_message(response['message']['content'], 'assistant'))
    return response


def main():
    while True:
        user_input = input("Enter your question (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break

        response = ask(user_input)

        if 'message' in response and 'tool_calls' in response['message'] and response['message']['tool_calls']:            
            tool_calls = response['message']['tool_calls']
            #tool_results = []  # Collect all tool results

            for tool_call in tool_calls:
                tool_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']

                if tool_name == 'get_current_weather' and 'city' in arguments:
                    result = get_current_weather(arguments['city'])
                    chat_messages.append(create_message(result, "tool"))
                    print("Weather function result:", result)
                elif tool_name == 'what_is_bigger' and 'n' in arguments and 'm' in arguments:
                    n, m = float(arguments['n']), float(arguments['m'])
                    result = what_is_bigger(n, m)
                    chat_messages.append(create_message(result, "tool"))
                    print("Comparison function result:", result)
                else:
                    result = "No valid arguments found for function: " + tool_name
                    chat_messages.append(create_message(result, "tool"))
                    print(result)

                    # Return result of all tools used to LLM
            response_after_tools = ollama.chat(
                model=model,
                messages=chat_messages
            )
            chat_messages.append(create_message(response_after_tools['message']['content'], 'assistant'))
            print("AI response:", response_after_tools['message']['content'])

        else:
            # If no tool calls or no valid arguments, use the LLM's response
            print("AI response:", response['message']['content'])


if __name__ == "__main__":
    main()