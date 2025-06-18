# Reinforcement Learning for Prompt Generation

The purpose of this repo, is to fine tune a model instance and increase its performance over time via human in the loop training. 

## Steps To Run this Project

- Make sure you have python version > 3.10 installed. (If it is a stable version, it is better).
- To download the required dependencies, in your project root, run the requirements.txt file. You can do so with the following command : 
        ```bash
        pip install -r requirements.txt
        ```
- Navigate to the env file, and update the OPEN AI / AZURE OPEN AI keys and endpoints based on your subscription.
- Set the training environment to openai or azure depending on if you are running the model on openai or azure.
- You can also swap/update the hyper parameters for testing and training the model. 


## Usage

Navigate to the root directory and execute the following code: 

- For running the main chat interface run : 
     ```bash
         python chat.py
     ```
- In another terminal instance run: 
     ```bash
         python feedback.py
     ```


Note that at times, python is not recognized. Try with py and python3 is needed in this case. Further, the only reason we are running the feedback loop as an API is to have it execute fast and asynchronously. It is responsive this way. 


Lastly, to execute the HITL training process, execute: 

 ```bash
        python hitl.py 
 ```

This will fine tune the existing model with the added suggestions / feedback to better performance. 