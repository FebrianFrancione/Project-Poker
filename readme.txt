Key files:

Requirements (besides ML packages used during the course):
pypokerengine
pypokerGUI


Files:
-RLPokerPlayer.py
This is where the Reinforcement learning player is implemented, all major strategies and NN's are implemented in object oriented fashion
Hyper parameters belong to the RL Player class.
-BluffingPlayer.py 
AI Player that will raise no matter what, unless it can't then it will fold
-HonestPlayer.py 
AI Player that will bet its hand according to a simulated win rate and number of other players
-GameLoop.py 
Main game loop with plotting functions, you can run this file to run the experiment or use the jupyter notebook
-poker_conf.yaml
Used for pokerGUI, not necessary but used to show a video during the video presentation


Running the Experiment:

Open the jupyter notebook GameLoop.ipynb, run this code in this file to run the experiment
Training took approximately 10 minutes on my machine for each scenario
Due to an unknown bug it is better to run each scenario at a time and then rerun the experiment by uncommenting which scenario you want to run

Reinforcement Learning player code heavily influenced from the Cart Pole example in the video tutorial series by DeepLizard found here:
https://deeplizard.com/learn/video/ZaILVnqZFCg
We modified it for our use case

to use the PokerGUI to play you can run the following in the terminal:
pypokergui serve poker_conf.yaml --port 8000 --speed fast

this will set up an interactive webpage to play poker against 3 RL players