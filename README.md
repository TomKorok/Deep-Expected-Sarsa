# Deep-Expected-Sarsa
Deep expected Sarsa algorithm which was created for a university project. 
It runs using the gym environment, especially the Breakout Atari game.
The algorithm was created based on the following tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html.
But the Q-learning update method was changed to the Sarsa update method which resulted in a significant performance increase.

The deep_expected_sarsa.py train the model while the env_test.py file display the game so the performance can be observed.
The deep_expected_sarsa.py file can save the trained model which later can be used by the env_test.py file.

To run the project python 3.11 should be used.
