# cs221project
Emily, Jon and Zane's 221 final project


Log Post Progress Report
- brainstorm session. first tasks:
    - better following and leading
    - A* working
    - multiple boids
- better following / leading
    - make leading not in a circle
    - make leader bounce off walls
    - POSSIBLE EXTENSION: change velocity as a possible action
- Adding multiple boids
    - issue … how do we train multiple ones at the same time …. do we want them to have different learned policies or the same?
    - in an ideal world, they have the same optimal policy
    - IDEA: train one follower, give that policy to all of the followers, then retrain a follower based on the other followers as well, then apply that learned policy to all and repeat until convergence
