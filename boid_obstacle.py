#!/usr/bin/env python
# Boid implementation in Python using PyGame
# Ben Dowling - www.coderholic.com

import sys, pygame, random, math, os
import QLearnBoid
import QLearnBoidO
import search_util
from collections import defaultdict
from QLearnBoid import QLearnBoid
from QLearnBoidO import QLearnBoidObstacles
from QLearnBoid import followLeaderBoidFeatureExtractorV2, distance
from QLearnBoidO import followLeaderBoidFeatureExtractorObstacles

pygame.init()

size = width, height = 1000, 600

def defineMaze(buffer):
    obstacles = defaultdict(bool)
    for r in range(1000):
        for c in range(600):
            obstacles[(r,c)] = False

    for r in range(100 - buffer, 100 + buffer):
        for c in range(0, 400 + buffer):
            obstacles[(r,c)] = True

    for r in range(200 - buffer, 200 + buffer):
        for c in range(0, 100 + buffer):
            obstacles[(r,c)] = True
        for c in range(200, 400):
            obstacles[(r,c)] = True

    for r in range(300 - buffer, 300 + buffer):
        for c in range(200 - buffer, 300 + buffer):
            obstacles[(r,c)] = True

    for r in range(400 - buffer, 400 + buffer):
        for c in range(100 - buffer, 300 + buffer):
            obstacles[(r,c)] = True

    for r in range(500 - buffer, 500 + buffer):
        for c in range(100 - buffer, 500 + buffer):
            obstacles[(r,c)] = True

    for r in range(600 - buffer, 600 + buffer):
        for c in range(100 - buffer, 600 + buffer):
            obstacles[(r,c)] = True

    for r in range(700 - buffer, 700 + buffer):
        for c in range(100 - buffer, 300 + buffer):
            obstacles[(r,c)] = True

    for r in range(800 - buffer, 800 + buffer):
        for c in range(300 - buffer, 500 + buffer):
            obstacles[(r,c)] = True

    for r in range(800 - buffer, 800 + buffer):
        for c in range(0 - buffer, 200 + buffer):
            obstacles[(r,c)] = True

    for r in range(900 - buffer, 900 + buffer):
        for c in range(100 - buffer, 1000 + buffer):
            obstacles[(r,c)] = True

    for r in range(200 - buffer, 400 + buffer):
        for c in range(100 - buffer, 100 + buffer):
            obstacles[(r,c)] = True

    for r in range(500 - buffer, 600 + buffer):
        for c in range(100 - buffer, 100 + buffer):
            obstacles[(r,c)] = True

    for r in range(200 - buffer, 500 + buffer):
        for c in range(400 - buffer, 400 + buffer):
            obstacles[(r,c)] = True

    for r in range(100 - buffer, 500 + buffer):
        for c in range(500 - buffer, 500 + buffer):
            obstacles[(r,c)] = True

    for r in range(700 - buffer, 800 + buffer):
        for c in range(300 - buffer, 300 + buffer):
            obstacles[(r,c)] = True

    for r in range(700 - buffer, 900 + buffer):
        for c in range(500 - buffer, 500 + buffer):
            obstacles[(r,c)] = True
    return obstacles

def defineMaze2(buffer):
    obstacles = defaultdict(bool)
    for r in range(width):
        for c in range(height):
            obstacles[(r,c)] = False

    for r in range(100 - buffer, 100 + buffer):
        for c in range(0, 300 + buffer):
            obstacles[(r,c)] = True
        for c in range(400 - buffer, 500 + buffer):
            obstacles[(r,c)] = True

    for r in range(300 - buffer, 300 + buffer):
        for c in range(100 - buffer, 300 + buffer):
            obstacles[(r,c)] = True

    for r in range(400 - buffer, 400 + buffer):
        for c in range(100 - buffer, 300 + buffer):
            obstacles[(r,c)] = True
        for c in range(400 - buffer, 500 + buffer):
            obstacles[(r,c)] = True

    for r in range(500 - buffer, 500 + buffer):
        for c in range(100 - buffer, 400 + buffer):
            obstacles[(r,c)] = True

    for r in range(600 - buffer, 600 + buffer):
        for c in range(0, 300 + buffer):
            obstacles[(r,c)] = True
        for c in range(550 - buffer, 600):
            obstacles[(r,c)] = True

    for r in range(700 - buffer, 700 + buffer):
        for c in range(300 - buffer, 450 + buffer):
            obstacles[(r,c)] = True

    for r in range(900 - buffer, 900 + buffer):
        for c in range(150 - buffer, 1000):
            obstacles[(r,c)] = True

    for r in range(200 - buffer, 300 + buffer):
        for c in range(100 - buffer, 100 + buffer):
            obstacles[(r,c)] = True

    for r in range(400 - buffer, 500 + buffer):
        for c in range(100 - buffer, 100 + buffer):
            obstacles[(r,c)] = True

    for r in range(700 - buffer, 900 + buffer):
        for c in range(150 - buffer, 150 + buffer):
            obstacles[(r,c)] = True

    for r in range(600 - buffer, 700 + buffer):
        for c in range(300 - buffer, 300 + buffer):
            obstacles[(r,c)] = True

    for r in range(300 - buffer, 400 + buffer):
        for c in range(300 - buffer, 300 + buffer):
            obstacles[(r,c)] = True

    for r in range(100 - buffer, 700 + buffer):
        for c in range(400 - buffer, 400 + buffer):
            obstacles[(r,c)] = True
    return obstacles

obstacles = defineMaze(12)
fat_obstacles = defineMaze(30)
black = 0, 0, 0
white = 255, 255, 255
minvel, maxvel = 0, 3

maxVelocity = 4
numBoids = 0
boids = []

crashdistance = 30


leader_exists = True

if leader_exists:
    numBoids += 1

class Boid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocityX = random.randint(1, 10) / 10.0
        self.velocityY = random.randint(1, 10) / 10.0
        self.angle = 0

    "Return the distance from another boid"
    def distance(self, boid):
        distX = self.x - boid.x
        distY = self.y - boid.y        
        return math.sqrt(distX * distX + distY * distY)

    "Move closer to a set of boids"
    def moveCloser(self, boids):
        if len(boids) < 1: return
            
        # calculate the average distances from the other boids
        avgX = 0
        avgY = 0
        for boid in boids:
            if boid.x == self.x and boid.y == self.y:
                continue
                
            avgX += (self.x - boid.x)
            avgY += (self.y - boid.y)

        avgX /= len(boids)
        avgY /= len(boids)

        # set our velocity towards the others
        distance = math.sqrt((avgX * avgX) + (avgY * avgY)) * -1.0
       
        self.velocityX -= (avgX / 100.0)
        self.velocityY -= (avgY / 100.0)
        
    "Move with a set of boids"
    def moveWith(self, boids):
        if len(boids) == 0: return
        # calculate the average velocities of the other boids
        avgX = 0
        avgY = 0
                
        for boid in boids:
            avgX += boid.velocityX
            avgY += boid.velocityY

        avgX /= len(boids)
        avgY /= len(boids)

        # set our velocity towards the others
        self.velocityX += (avgX / 8)
        self.velocityY += (avgY / 8)
    
    "Move away from a set of boids. This avoids crowding"
    def moveAway(self, boids, minDistance):
        if len(boids) < 1: return
        
        distanceX = 0
        distanceY = 0
        numClose = 0

        for boid in boids:
            distance = self.distance(boid)
            if  distance < minDistance:
                numClose += 1
                xdiff = (self.x - boid.x) 
                ydiff = (self.y - boid.y) 
                
                if xdiff >= 0: xdiff = math.sqrt(minDistance) - xdiff
                elif xdiff < 0: xdiff = -math.sqrt(minDistance) - xdiff
                
                if ydiff >= 0: ydiff = math.sqrt(minDistance) - ydiff
                elif ydiff < 0: ydiff = -math.sqrt(minDistance) - ydiff

                distanceX += xdiff 
                distanceY += ydiff 
        
        if numClose == 0:
            return
            
        self.velocityX -= distanceX / 5
        self.velocityY -= distanceY / 5
        
    "Perform actual movement based on our velocity"
    def move(self):
        if abs(self.velocityX) > maxVelocity or abs(self.velocityY) > maxVelocity:
            scaleFactor = maxVelocity / max(abs(self.velocityX), abs(self.velocityY))
            self.velocityX *= scaleFactor
            self.velocityY *= scaleFactor
        
        self.x += self.velocityX
        self.y += self.velocityY

        # Try wrap around - may need variable width and hieght
        #self.x = (self.x + width) % width
        #self.y = (self.y + height) % height


class LeadBoid(Boid):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.speed = 2
        self.angle = 0.0
        self.direction = [0, 0]
        self.stepCounts = 0
        self.multiple = 1
        self.velocityX = random.uniform(-1,1)
        self.velocityY = random.uniform(-1,1)

    "Move closer to a set of boids"
    def moveCloser(self, boids):
        return
        
    "Move with a set of boids"
    def moveWith(self, boids):
        return
    
    "Move away from a set of boids. This avoids crowding"
    def moveAway(self, boids, minDistance):
        return
        
    "Perform actual movement based on our velocity"
    def move(self):

        if self.stepCounts == 20:
            self.multiple = random.uniform(.8,1.2)
            self.stepCounts = 0
            if self.angle == 0: 
                self.angle += 20
        else:
            self.multiple = 1
            self.stepCounts += 1


        self.angle *= self.multiple
        # self.angle += 1
        self.angle = self.angle % 360
        self.direction[0] = math.sin(-math.radians(self.angle))
        self.direction[1] = -math.cos(math.radians(self.angle))

        # calculate the position from the direction and speed
        self.x += self.direction[0]*self.speed
        self.y += self.direction[1]*self.speed

        if self.x <= 10 or self.x >= width - 10:
            self.angle += 90
        if self.y <= 10 or self.y >= height - 10:
            self.angle += 90

class StraightLineBoid(Boid):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # move across the screen
        self.speed = 3
        self.angle = 90
        self.direction = [1, 0]

    "Move closer to a set of boids"
    def moveCloser(self, boids):
        return
        
    "Move with a set of boids"
    def moveWith(self, boids):
        return
    
    "Move away from a set of boids. This avoids crowding"
    def moveAway(self, boids, minDistance):
        return
        
    "Perform actual movement based on our velocity"
    def move(self):
        #self.velocityX += random.uniform(-1,1)
        #self.velocityY += random.uniform(-1,1)
        #if self.velocityX**2 + self.velocityY**2 > maxVelocity**2:
            #scaleFactor = maxVelocity / max(abs(self.velocityX), abs(self.velocityY))
            #self.velocityX *= scaleFactor
            #self.velocityY *= scaleFactor
        self.direction[0] = math.sin(math.radians(self.angle))
        self.direction[1] = -math.cos(math.radians(self.angle))

        self.x += self.direction[0]*self.speed
        self.y += self.direction[1]*self.speed

        # Try wrap around - may need variable width and hieght
        #self.x = (self.x + width) % width
        #self.y = (self.y + height) % height

class LearningBoid():
    def __init__(self, x, y, angle = 0.0, speed = 2): 
        self.x = x
        self.y = y
        # move across the screen

        self.speed = speed
        self.angle = angle
        self.direction = [0, 0]

    def move(self, action):
        # Assume for now that an action is just an angle movement
        if action[0] != None:
            self.angle += action[0]
            self.speed += action[1]
            if self.speed > maxvel:
                self.speed = maxvel
            elif self.speed < minvel:
                self.speed = minvel
            self.angle = self.angle % 360
            self.direction[0] = math.sin(math.radians(self.angle))
            self.direction[1] = -math.cos(math.radians(self.angle))

            # calculate the position from the direction and speed
            self.x += self.direction[0]*self.speed
            self.y += self.direction[1]*self.speed

class CircleBoid(Boid):
    def __init__(self, x, y):
        self.x = x
        self.y = y

        self.speed = 3
        self.angle = 0.0
        self.direction = [0, 0]


    "Move closer to a set of boids"
    def moveCloser(self, boids):
        return
        
    "Move with a set of boids"
    def moveWith(self, boids):
        return
    
    "Move away from a set of boids. This avoids crowding"
    def moveAway(self, boids, minDistance):
        return
        
    "Perform actual movement based on our velocity"
    def move(self):
        # We want to just move at a constant 5 degree angle

        self.angle += 1
        self.angle = self.angle % 360
        self.direction[0] = math.sin(-math.radians(self.angle))
        self.direction[1] = -math.cos(math.radians(self.angle))

        # calculate the position from the direction and speed
        self.x += self.direction[0]*self.speed
        self.y += self.direction[1]*self.speed

class SearchBoid(Boid):
    def __init__(self):
        self.x = 50
        self.y = 50
        self.step = 0
        self.speed = 3
        self.angle = 0

        search = search_util.UniformCostSearch()
        search.solve(search_util.MazeProblem((50,50), ((950, 550)), fat_obstacles))
        self.actions = search.actions

    "Move closer to a set of boids"
    def moveCloser(self, boids):
        return
        
    "Move with a set of boids"
    def moveWith(self, boids):
        return
    
    "Move away from a set of boids. This avoids crowding"
    def moveAway(self, boids, minDistance):
        return
        
    "Perform actual movement based on our velocity"
    def move(self):
        # We want to just move at a constant 5 degree angle
        # calculate the position from the direction and speed
        if self.step < len(self.actions):
            oldx = self.x
            oldy = self.y
            self.x = self.actions[self.step][0]
            self.y = self.actions[self.step][1]
            self.step += self.speed
            if self.x - oldx == 0:
                self.angle = 90
                if self.y - oldy > 0:
                    self.angle = 270
            else:
                self.angle = math.degrees(math.atan((self.y - oldy)/(self.x - oldx)))


def test_maze(rl):

    # Super simple test right now with one leader and one 
    # follower controlled by the rl algorithm
    screen = pygame.display.set_mode(size)

    bird = pygame.image.load("bird.png")
    birdrect = bird.get_rect()
    lead = pygame.image.load("bird1.png")
    leadrect = lead.get_rect()

    #leaderBoid = StraightLineBoid(55, height / 2.0)
    leaderBoid = SearchBoid()
    #leaderBoid = LeadBoid(55, height / 2.0)
    # Define the start state for our rl algorithm
    #learnerBoid = LearningBoid(25, height / 2.0, 90)
    learnerBoid = LearningBoid(50, 25, 90)
    learnerBoid2 = LearningBoid(25, 50, 90)


    # Define the start state that will be passed to our learning algorithm
    state = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, leaderBoid.angle), leaderBoid.speed, (width, height))
    state2 = ((learnerBoid2.x, learnerBoid2.y, learnerBoid2.angle), (leaderBoid.x, leaderBoid.y, leaderBoid.angle), leaderBoid.speed, (width, height))

    background_surface = pygame.Surface((width, height))
    background_surface.fill(white)
    for key in obstacles:
        if obstacles[key]:
            background_surface.set_at(key, black)


    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        # Move both boids
        leaderBoid.move()

        action = rl.getAction(state)
        action2 = rl.getAction(state2)
        learnerBoid.move(action)
        learnerBoid2.move(action2)

        # Calculate the new state
        newState = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, learnerBoid.angle), leaderBoid.speed, (width, height))
        newState2 = ((learnerBoid2.x, learnerBoid2.y, learnerBoid2.angle), (leaderBoid.x, leaderBoid.y, learnerBoid.angle), leaderBoid.speed, (width, height))
    
        #show the maze
        screen.blit(background_surface, (0,0))

        state = newState
        state2 = newState2
        # Draw the boids
        # Draw the leader
        boidRect = pygame.Rect(leadrect)
        boidRect.x = leaderBoid.x
        boidRect.y = leaderBoid.y
        screen.blit(bird, boidRect)

        # Draw the learner
        boidRect = pygame.Rect(birdrect)
        boidRect.x = learnerBoid.x
        boidRect.y = learnerBoid.y
        screen.blit(bird, boidRect)

        boidRect = pygame.Rect(birdrect)
        boidRect.x = learnerBoid2.x
        boidRect.y = learnerBoid2.y
        screen.blit(bird, boidRect)
        
        pygame.display.flip()
        pygame.time.delay(1)

def allGood(x, y):
    poss = [0, 20]
    x = int(x)
    y = int(y)
    for x1 in poss:
        for y1 in poss:
            if (x + x1, y + y1) in obstacles:
                if obstacles[(x + x1,y + y1)]:
                    return False
                else: 
                    continue
            else:
                return False
    return True
# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate_maze(rl, numTrials=100, maxIterations=1200, verbose=False,
             sort=False, learning=True, qlearn=True):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

    def isfollowing(state):
        # Define following as being within 20-30 units
        #if not(distance(state[0], state[1]) > 20 and distance(state[0], state[1]) <= 35):
            #print distance(state[0], state[1])
        return distance(state[0], state[1]) > 20 and distance(state[0], state[1]) <= 35

    def reward(prev_state, new_state):
        # We will primarily calculate initial reward 
        # based on distance between leader and the flying bird

        # Calculate the previous distance
        old_learner_loc = prev_state[0]
        old_leader_loc = prev_state[1]
        distance_old = distance(old_learner_loc, old_leader_loc)

        # Calculate new distance 
        new_learner_loc = newState[0]
        new_leader_loc = newState[1]
        distance_new = distance(new_learner_loc, new_leader_loc)

        reward = 0
        # Base reward on how the distance changes

        if distance_new < crashdistance:
            reward = - 2*(1/distance_new)
        elif distance_old > distance_new:
            reward = distance_new / float(8)
            # reward += (1/distance_new)
        elif distance_old < distance_new:
            reward = - distance_new / float(2)
            # reward -= (1/distance_new)



        if not allGood(new_learner_loc[0], new_learner_loc[1]):
            return -1000

        return reward


    totalRewards = []  # The rewards we get on each trial
    following = []
    for trial in range(numTrials):
        # We want to start doing the simulation
        # Let us start by placing down a the leader and
        # the learning follower
        #leaderBoid = StraightLineBoid(55, height / 2.0)
        #leaderBoid = LeadBoid(55, height / 2.0)
        leaderBoid = SearchBoid()
        # Define the start state for our rl algorithm
        #learnerBoid = LearningBoid(25, height / 2.0, 90)
        learnerBoid = LearningBoid(50, 50, 90)

        if not qlearn:
            learnerBoid = Boid(50, 50)

        # Define the start state that will be passed to our learning algorithm
        state = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, 0), leaderBoid.speed, (width, height))

        # We have to define the start state. We should start the bird close to the
        # follow bird
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        time_steps_following = 0
        for _ in range(maxIterations):
            if qlearn:
                # Get the action predicted by the bird learning algorithm
                action = rl.getAction(state)
                # Move the learning bird
                learnerBoid.move(action)
            else:
                learnerBoid.move()

            # Move the leading bird
            leaderBoid.move()

            newState = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, 0), leaderBoid.speed, (width, height))
            
            reward1 = reward(state, newState)

            if qlearn:
                sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)


            if learning:
                rl.incorporateFeedback(state, action, reward1, newState)

            totalReward += totalDiscount * reward1
            if trial % 3 == 0:
                time_steps_following += 1 if isfollowing(newState) else 0

            state = newState
        if verbose:
            print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        if trial % 3 == 0:
            following.append(time_steps_following)
            rl.printWeights()
        totalRewards.append(totalReward)
    return totalRewards, following

## Run the game!
# Define the actions for the boids
# as the angles that can turn
def actions(state):
    angles = [-45, -35, -20, -10, -5, -2, 0, 2, 5, 10, 20, 35, 45, 180]
    velocities = [-.3, -.2, -.1, 0, .1, .2, .3]

    toReturn = []

    for angle in angles:
        for speed in velocities:
            toReturn.append((angle, speed))

    return toReturn
    #return [None, -45, 0, 45, 90, -90, 135, -135, 180]

rl = QLearnBoidObstacles(actions, 0.05, followLeaderBoidFeatureExtractorObstacles, obstacles)
if False:
    results, following = simulate_maze(rl)
    print following
    rl.printWeights()
else:
    rl.weights = {'too-close': 1.4091332158627652, 'distance': -210.27940631153584, 'distance-delta': -43.32266379857211, 'very_nearby': -1007.8269221679548}

#total_rewards = simulate_fixed(rl)
print "***total rewards for this different simulations***"
#print total_rewards
rl.explorationProb = 0
print "--we got here---"

# results, following = simulate_maze(rl, numTrials=20, learning=False)
# results1, following1 = simulate_maze(rl, numTrials=20, learning=False, qlearn=False)
# print results 
# print following
# print'---normal boid---'
# print results1
# print following1

# os.system('say "your program has finished"')
# os.system('say "you should really switch over to that window"')
# os.system('say "3"')
# os.system('say "2"')
# os.system('say "1"')
test_maze(rl)
   