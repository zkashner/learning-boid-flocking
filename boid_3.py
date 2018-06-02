#!/usr/bin/env python
# Boid implementation in Python using PyGame
# Ben Dowling - www.coderholic.com

import sys, pygame, random, math
import QLearnBoid
from QLearnBoid import QLearnBoid
from QLearnBoid import followLeaderBoidFeatureExtractorV2, distance

pygame.init()

size = width, height = 1000, 600
black = 0, 0, 0
white = 255, 255, 255

maxVelocity = 4
numBoids = 0
boids = []

crashdistance = 20


leader_exists = True

if leader_exists:
    numBoids += 1

class Boid:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.velocityX = random.randint(1, 10) / 10.0
        self.velocityY = random.randint(1, 10) / 10.0

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


        # self.velocityX += random.uniform(-1,1)
        # self.velocityY += random.uniform(-1,1)
        # if self.velocityX**2 + self.velocityY**2 > maxVelocity**2:
        #     scaleFactor = maxVelocity / max(abs(self.velocityX), abs(self.velocityY))
        #     self.velocityX *= scaleFactor
        #     self.velocityY *= scaleFactor
        
        # self.x += self.velocityX
        # self.y += self.velocityY

        
       
        # Try wrap around - may need variable width and hieght
        #self.x = (self.x + width) % width
        #self.y = (self.y + height) % height

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
        if action != None:
            self.angle += action
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


'''
screen = pygame.display.set_mode(size)

bird = pygame.image.load("bird.png")
birdrect = bird.get_rect()
lead = pygame.image.load("bird1.png")
leadrect = lead.get_rect()

# create boids at random positions
for i in range(numBoids - 1):
    boids.append(Boid(random.randint(0, width), random.randint(0, height)))

if leader_exists:
    #boids.append(LeadBoid(random.randint(0, width), random.randint(0, height))) 
    boids.append(straightLineBoid(30, height / 2.0)) 
    #boids.append(circleBoid(500, 300))


while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    for boid in boids:
        closeBoids = []
        for otherBoid in boids:
            if otherBoid == boid: continue
            distance = boid.distance(otherBoid)
            if distance < 100:
                closeBoids.append(otherBoid)

        
        boid.moveCloser(closeBoids)
        boid.moveWith(closeBoids)  
        boid.moveAway(closeBoids, 30)

        # ensure they stay within the screen space
        # if we roubound we can lose some of our velocity
        
        border = 25
        if boid.x < border and boid.velocityX < 0:
            boid.velocityX = -boid.velocityX * random.random()
        if boid.x > width - border and boid.velocityX > 0:
            boid.velocityX = -boid.velocityX * random.random()
        if boid.y < border and boid.velocityY < 0:
            boid.velocityY = -boid.velocityY * random.random()
        if boid.y > height - border and boid.velocityY > 0:
            boid.velocityY = -boid.velocityY * random.random()
        
            
        boid.move()
        
    screen.fill(white)
    for i in range(len(boids)):
        boid = boids[i]
        if leader_exists and i == len(boids) - 1:
            boidRect = pygame.Rect(leadrect)
            boidRect.x = boid.x
            boidRect.y = boid.y
            screen.blit(lead, boidRect)
        else:
            boidRect = pygame.Rect(birdrect)
            boidRect.x = boid.x
            boidRect.y = boid.y
            screen.blit(bird, boidRect)
    pygame.display.flip()
    pygame.time.delay(1)
'''

def test_rl(rl):
    # Super simple test right now with one leader and one 
    # follower controlled by the rl algorithm
    screen = pygame.display.set_mode(size)

    bird = pygame.image.load("bird.png")
    birdrect = bird.get_rect()
    lead = pygame.image.load("bird1.png")
    leadrect = lead.get_rect()

    #leaderBoid = StraightLineBoid(55, height / 2.0)
    leaderBoid = LeadBoid(500, 300)
    #leaderBoid = LeadBoid(55, height / 2.0)
    # Define the start state for our rl algorithm
    #learnerBoid = LearningBoid(25, height / 2.0, 90)
    learnedBoids = []
    learnedBoids.append(LearningBoid(450, 300, 90))
    learnedBoids.append(LearningBoid(350, 310, 90))
    learnedBoids.append(LearningBoid(575, 350, 90))
    learnedBoids.append(LearningBoid(650, 400, 90))
    #learnerBoid = LearningBoid(450, 300, 90)
    #learnerBoid2 = LearningBoid(350, 310, 90)

    # Define the start states that will be passed to our learning algorithm
    states = []
    for boid in learnedBoids:
        states.append(((boid.x, boid.y, boid.angle), (leaderBoid.x, leaderBoid.y, leaderBoid.angle), leaderBoid.speed, (width, height)))
    
    #state = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, learnerBoid.angle), leaderBoid.speed, (width, height))
    #state2 = ((learnerBoid2.x, learnerBoid2.y, learnerBoid2.angle), (leaderBoid.x, leaderBoid.y, learnerBoid.angle), leaderBoid.speed, (width, height))


    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        # Move both boids
        leaderBoid.move()

        # Move the followers
        for i in range(len(learnedBoids)):
            action = rl.getAction(states[i])
            learnedBoids[i].move(action)
        
        #action = rl.getAction(state)
        #action2 = rl.getAction(state2)
        #learnerBoid.move(action)
        #learnerBoid2.move(action2)

        # Calculate the new states
        for i in range(len(states)):
            states[i] = ((learnedBoids[i].x, learnedBoids[i].y, learnedBoids[i].angle), (leaderBoid.x, leaderBoid.y, leaderBoid.angle), leaderBoid.speed, (width, height))
        
        #newState = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, learnerBoid.angle), leaderBoid.speed, (width, height))
        #newState2 = ((learnerBoid2.x, learnerBoid2.y, learnerBoid2.angle), (leaderBoid.x, leaderBoid.y, learnerBoid.angle), leaderBoid.speed, (width, height))
    
        screen.fill(white)

        #state = newState
        #state2 = newState2

        # Draw the boids
        # Draw the leader
        boidRect = pygame.Rect(leadrect)
        boidRect.x = leaderBoid.x
        boidRect.y = leaderBoid.y
        screen.blit(lead, boidRect)

        # Draw the learners
        for boid in learnedBoids:
            boidRect = pygame.Rect(birdrect)
            boidRect.x = boid.x
            boidRect.y = boid.y
            screen.blit(bird, boidRect)

        #boidRect = pygame.Rect(birdrect)
        #boidRect.x = learnerBoid2.x
        #boidRect.y = learnerBoid2.y
        #screen.blit(bird, boidRect)
        
        pygame.display.flip()
        pygame.time.delay(1)
   

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(rl, numTrials=45, maxIterations=5000, verbose=False,
             sort=False):
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
            #reward = -110
            if distance_old <= distance_new:
                reward = 35
            else:
                reward = -45
        elif distance_old > distance_new:
            #reward = 400
            reward = 5
        elif distance_old < distance_new:
            #if distance_new > 35:
                #reward = -35
            #reward = -150
            #else:
            reward = -30
        #elif distance_new > 35:
            #reward = -35
        #else:
            #reward = 1


        #if new_learner_loc[0] < 0 or new_learner_loc[0] > width:
            #reward += -100 + min(new_learner_loc[0], width - new_learner_loc[0])
        #if new_learner_loc[1] < 0 or new_learner_loc[1] > height:
            #reward += -100 + min(new_learner_loc[1], height - new_learner_loc[1])

        return reward

    totalRewards = []  # The rewards we get on each trial
    following = []
    for trial in range(numTrials):
        # We want to start doing the simulation
        # Let us start by placing down a the leader and
        # the learning follower
        #leaderBoid = StraightLineBoid(55, height / 2.0)
        #leaderBoid = LeadBoid(55, height / 2.0)
        leaderBoid = LeadBoid(500, 300)
        # Define the start state for our rl algorithm
        #learnerBoid = LearningBoid(25, height / 2.0, 90)
        learnerBoid = LearningBoid(450, 300, 90)

        # Define the start state that will be passed to our learning algorithm
        state = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, leaderBoid.angle), leaderBoid.speed, (width, height))

        # We have to define the start state. We should start the bird close to the
        # follow bird
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        time_steps_following = 0
        for _ in range(maxIterations):
            # Get the action predicted by the bird learning algorithm
            action = rl.getAction(state)
            # Move the learning bird
            learnerBoid.move(action)

            # Move the leading bird
            leaderBoid.move()

            newState = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, leaderBoid.angle), leaderBoid.speed, (width, height))
            
            reward1 = reward(state, newState)

            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

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

def simulate_fixed(rl, numTrials=10, maxIterations=1000, verbose=False,
             sort=False):
    # Return i in [0, ..., len(probs)-1] with probability probs[i].
    def sample(probs):
        target = random.random()
        accum = 0
        for i, prob in enumerate(probs):
            accum += prob
            if accum >= target: return i
        raise Exception("Invalid probs: %s" % probs)

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
            reward = -110
        elif distance_old > distance_new:
            reward = 400
        elif distance_old < distance_new:
            reward = -150


        if new_learner_loc[0] < 0 or new_learner_loc[0] > width:
            reward += -100 + min(new_learner_loc[0], width - new_learner_loc[0])
        if new_learner_loc[1] < 0 or new_learner_loc[1] > height:
            reward += -100 + min(new_learner_loc[1], height - new_learner_loc[1])

        return reward


    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        # We want to start doing the simulation
        # Let us start by placing down a the leader and
        # the learning follower
        leaderBoid = StraightLineBoid(55, height / 2.0)
        # Define the start state for our rl algorithm
        learnerBoid = LearningBoid(25, height / 2.0, 90)

        # Define the start state that will be passed to our learning algorithm
        state = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, learnerBoid.angle), leaderBoid.speed, (width, height))

        # We have to define the start state. We should start the bird close to the
        # follow bird
        sequence = [state]
        totalDiscount = 1
        totalReward = 0
        for _ in range(maxIterations):
            # Get the action predicted by the bird learning algorithm
            action = rl.getAction(state)
            # Move the learning bird
            learnerBoid.move(action)

            # Move the leading bird
            leaderBoid.move()

            newState = ((learnerBoid.x, learnerBoid.y, learnerBoid.angle), (leaderBoid.x, leaderBoid.y, leaderBoid.angle), leaderBoid.speed, (width, height))
            
            reward1 = reward(state, newState)

            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            totalReward += totalDiscount * reward1

            state = newState

        totalRewards.append(totalReward)
    return totalRewards



## Run the game!
# Define the actions for the boids
# as the angles that can turn
def actions(state):
    return [None, -45, -35, -20, -10, -5, -2, 0, 2, 5, 10, 20, 35, 45]
    #return [None, -45, 0, 45, 90, -90, 135, -135, 180]

rl = QLearnBoid(actions, 0.05, followLeaderBoidFeatureExtractorV2)
results, following = simulate(rl)
print following
rl.printWeights()
#total_rewards = simulate_fixed(rl)
print "***total rewards for this different simulations***"
#print total_rewards
rl.explorationProb = 0
test_rl(rl)
