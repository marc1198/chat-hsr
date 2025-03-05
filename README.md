# IDRA-H

Intelligent Dialogue for Robot Applications with Humans: a system pipeline for intutitive dialogue between a user and a robot through the invovlement of LLMs.

## How to run on Toyota HSR (with-ROS)

1. Make sure the state machine is running and the HSR is connected.
2. Before running the code, make sure to adjust the shell script. 
3. Source the shell script.

```
source ros_entrypoint.sh 
```

4. Start roslaunch server.
```
roslaunch launch/with-ROS__as.launch
```

5. Start your experience with IDRA-H.

NOTE: The interaction with the HSR pipeline can also be adjused to run with models through Ollama.


## How to run only the dialogue, without the connection to the robot (without-ROS)
### Run with models like LLama 3.1, Gemma2 & Qwen2.5 through Ollama

1. If required, adjust the `model` in main_single_agents.py.

2. Make sure Ollama is running with the desired model.

```
docker start <your_ollama_container>
docker exec -it <your ollama_container> bash
```

3. Run the code and start your dialogue
```
python3 main_single_agents.py
```
