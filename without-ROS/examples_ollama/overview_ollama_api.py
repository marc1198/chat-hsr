#region 1. Chat Response without streaming
"""import ollama

# Setting up the model, enabling streaming responses, and defining the input messages
ollama_response = ollama.chat(model='llama3.1', messages=[
   {
     'role': 'system',
     'content': 'You are a helpful assistant with sound philosophical knowledge.',
   },
   {
     'role': 'user',
     'content': 'Explain to me the meaning of life?',
   },
])
# Printing out of the generated response
print(ollama_response['message']['content'])"""
#endregion

#region 2. Chat response with streaming
"""import ollama

# Setting up the model, enabling streaming responses, and defining the input messages
ollama_response = ollama.chat(
  model='llama3.1',
  stream=True,
  messages=[
    {
      'role': 'system',
      'content': 'You are a helpful assistant with sound philosophical knowledge.',
    },
    {
      'role': 'user',
      'content': 'Explain to me the meaning of life?',
    },
  ]
)

# Printing out each piece of the generated response while preserving order
for chunk in ollama_response:
  print(chunk['message']['content'], end='', flush=True)"""
#endregion

#region 3. Chat implementation with history - hardcoded chat
"""import ollama

# Initializing an empty list for storing the chat messages and setting up the initial system message
chat_messages = []
system_message='You are a helpful assistant.'

# Defining a function to create new messages with specified roles ('user' or 'assistant')
def create_message(message, role):
  return {
    'role': role,
    'content': message
  }

# Starting the main conversation loop
def chat():
  # Calling the ollama API to get the assistant response
  ollama_response = ollama.chat(model='llama3.1', stream=True, options = {'temperature': 0}, messages=chat_messages)

  # Preparing the assistant message by concatenating all received chunks from the API
  assistant_message = ''
  for chunk in ollama_response:
    assistant_message += chunk['message']['content']
    print(chunk['message']['content'], end='', flush=True)
    
  # Adding the finalized assistant message to the chat log
  chat_messages.append(create_message(assistant_message, 'assistant'))

# Function for asking questions - appending user messages to the chat logs before starting the `chat()` function
def ask(message):
  chat_messages.append(
    create_message(message, 'user')
  )
  print(f'\n\n--{message}--\n\n')
  chat()

# Sending two example requests using the defined `ask()` function
ask('Please list the 20 largest cities in the world.')
ask('How many of the cities listed are in South America?')"""
#endregion 

#region 4. Chat implementation with history - flexible chat
import ollama

# Create Agents
#ollama.create(model='agent_smart_guy', from_='llama3.1', system="You are an agent answering questions with very smart and precise answers.")
#ollama.create(model='agent_stupid_guy', from_='llama3.1', system="You are an agent answering questions with very stupid and long answers without really answering the question.")

# Initializing an empty list for storing the chat messages and setting up the initial system message
chat_messages = []

# Defining a function to create new messages with specified roles ('user' or 'assistant')
def create_message(message, role):
  return {
    'role': role,
    'content': message
  }

# Starting the main conversation loop
def chat():
  # Calling the ollama API to get the assistant response
  ollama_response = ollama.chat(model='llama3.1', stream=True, options = {'temperature': 0}, messages=chat_messages)

  # Preparing the assistant message by concatenating all received chunks from the API
  assistant_message = ''
  for chunk in ollama_response:
    assistant_message += chunk['message']['content']
    print(chunk['message']['content'], end='', flush=True)
    
  # Adding the finalized assistant message to the chat log
  chat_messages.append(create_message(assistant_message, 'assistant'))

# Function for asking questions - appending user messages to the chat logs before starting the `chat()` function
def ask(message):
  chat_messages.append(
    create_message(message, 'user')
  )
  print(f'\n\n--{message}--\n\n')
  chat()

# Main Loop for continuous conversation 
def main():
    #Start-Sequence
    print("Hi, I am here to assist you with the objects on the table. Please tell me what you need.")

    while True:
        user_input = input("\n Enter your prompt: ")
        if not user_input:
            exit()
        
        print("\nSending prompt to LLM")
        ask(user_input)

if __name__ == "__main__":
    main()
#endregion
