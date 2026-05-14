## Program Description

This program is a simple simulator of an autonomous car, which is trained using the NEAT (NeuroEvolution of Augmenting Topologies) evolutionary algorithm. Here's how it works and how the following aspects are fulfilled:

1. **Simulation:**
   - The simulation takes place in a pygame window.
   - The car moves along the track and makes decisions based on data from radar sensors.
   - The program simulates the behavior of an autonomous vehicle on the track.

2. **Neural Networks and Generational Learning:**
   - Each car in the population is represented by a genome, which consists of a neural network.
   - The NEAT algorithm evolves the population of cars to determine the optimal approach to navigating the track.
   - Cars are evaluated based on the distance traveled and rewarded for covering more distance.
   - The population evolves in successive generations, with better-adapted cars having a greater chance of passing on their genetic traits to the next generation.

3. **Sensor Integration:**
   - The car has radar sensors that detect the distance to obstacles on the track.
   - The `radardetection` function calculates the distance to the nearest obstacle in different directions.

4. **Collision Avoidance:**
   - The `avoid_collision` function checks if the car collides with an obstacle on the track, based on the color of the pixel at a specified point.
   - In case of a collision detection, the car is disabled.

5. **Decision Processing:**
   - The car makes decisions based on data from the radars and passes them to the neural network.
   - The neural network calculates a score for each direction, and the car selects the direction with the highest score.
   - Decisions are made based on the activation of neurons in the neural network, and specific actions (turn left/right) depend on the network's outputs.

This program demonstrates the basic application of the NEAT algorithm in simulating autonomous vehicle behavior. Through training across generations, cars learn to avoid collisions and navigate the track in the most efficient way possible.
