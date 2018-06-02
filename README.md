# cs221project
Emily, Jon and Zane's 221 final project


LOG: 
- post progress report we had a brainstorm session
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
BAD THINGS
- velocity could not change, can’t slow down when you land on it
- trying to add more features and explosion
- realized that 1 for our too close weight doesn’t work
- [500, 452, 749, 702, 806, 716, 721, 750, 758, 695, 745, 796, 776, 811, 666]
- defaultdict(<type 'float'>, {'too-close': 1.2508097663487758, 'distance': -40.587907965647666, 'distance-delta': -0.8857119209030659})
- FINAL working weights: defaultdict(<type 'float'>, {'too-close': 1.088858490843417, 'distance': -30.491060232703628, 'distance-delta': -0.4550626671892882})

second baseline: they all follow the leader with the same behavior: works in regards to the leader but flocking birds overlap 
to do: make it so the birds do not overlap with each other (need to interact with each other)
approach: iterative learning 
- train one with the leader, apply that behavior to many, then train one with the whole flock, re-apply that behavior to many, possibly repeat this process one more time (in an ideal world repeat to convergence)

Measurement ideas: 
- separation: count crashes 
- alignment: get mean deviation from leader angle
- cohesion: mean distance from average point of flock

Issues: testing is difficult is difficult when you have to train a sufficient amount 

