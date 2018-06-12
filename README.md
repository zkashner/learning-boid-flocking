# Flocking Behavior: Multi-Agent Q-Learning, Obstacle Avoidance and Real-World Simulations
Emily Cohen, Jon Gomes Selman, and Zane Kashner's CS221 final project

## Introduction

Our work is inspired by the "Boid" artificial life research program, first proposed by Craig Reynolds 
in 1986. This model seeks to capture the complex dynamics of collective motion exhibited in nature 
by flocks of birds, schools of fish, and other gregarious animals. Through studying the dynamics of such 
models, we can illuminate and better understand many complex animal behaviors, which can be directly 
applied to different real-world settings including dynamic multi-agent drones, robot swarms, quad-copters, 
and other robot systems. 

Although drone swarms are typically associated with military activity, there are a wide range of applications for 
collective task completion. For example, collective drone swarms can be used to monitor landscapes, such as farms, as well as survey war and disaster torn areas that would otherwise be inaccessible. 
Additionally, drones swarm can be used to deliver food, medicine, and materials to remote or hard to reach areas without compromising the safety of more people. A lab at MIT has proposed using
sailing robot swarms to clean up oil spills at a accelerated rate compared to current techniques. Overall, it is clear that the collective power of a group greatly outperforms that of an individual and opens the potential for solving many real world problems.  

The goal of our project is to better understand the dynamics of these groups with the hope of being able to apply what we learn to increase the efficiency and feasibility of utilizing robot swarms to help better our surroundings in ways that humans are not able to physically accomplish. Based on the Boid artificial life program for modeling flocking behavior, we propose to use learning techniques to learn this flocking behavior. Rather than using traditional rule based models to dictate the collective motion of flocking agents, we propose different reinforcement learning techniques to learn this complex task. We then hope to further extend our model to capture more complex flocking behaviors, such as obstacle avoidance, ultimately being able to apply our learned flocking behavior to a series of complex tasks such as path following through obstacle mazes.

The central challenge we face in this endeavor is fully defining an unstructured task. We are tasking
with determining relevant features for learning, as well as defining expressive reward functions to 
properly shape the learning process. In order to evaluate our results, we experiment with different 
self defined metrics to measure the success of our flocking algorithm compared to standard approaches. 
Lastly, for training and testing our learning algorithms we present a simulation environment to visualize
agent interaction dynamics. 

## This Repository

The code contained in this repository is broken into two parts, with a lot of recycling between the two of them.

### Flocking Behavior
In order to run the code around this call `python boid_3.py`. Additionally, make sure that the packages in the header are imported. This can all be done using `pip`

## Obstacle Manipulation
In order to run the boid through a maze call `python boid_obstacle.py`. For this, if you wish to train the weights of the Q-learned function approximation yourself, you can change the boolean at the bottom of the file to `True` and rather than using the weights we found, can determine these weights on your own.
