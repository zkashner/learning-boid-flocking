#!/usr/bin/env python
# Boid implementation in Python using PyGame
# Ben Dowling - www.coderholic.com

import sys, pygame, random, math

pygame.init()

size = width, height = 1000, 600
black = 0, 0, 0
white = 255, 255, 255

maxVelocity = 4
numBoids = 0
boids = []

crashdistance = 5


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
        self.velocityX = random.randint(1, 10) / 10.0
        self.velocityY = random.randint(1, 10) / 10.0

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
        self.velocityX += random.uniform(-1,1)
        self.velocityY += random.uniform(-1,1)
        if self.velocityX**2 + self.velocityY**2 > maxVelocity**2:
            scaleFactor = maxVelocity / max(abs(self.velocityX), abs(self.velocityY))
            self.velocityX *= scaleFactor
            self.velocityY *= scaleFactor
        
        self.x += self.velocityX
        self.y += self.velocityY

        # Try wrap around - may need variable width and hieght
        #self.x = (self.x + width) % width
        #self.y = (self.y + height) % height

class straightLineBoid(Boid):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # move across the screen
        self.speed = 3
        self.angle = 0
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

class learningBoid():
    def __init__(self, x, y, angle = 0.0, speed = 3): 
        self.x = x
        self.y = y
        # move across the screen

        self.speed = speed
        self.angle = angle
        self.direction = [0, 0]

    def move(self, action):
        # Assume for now that an action is just an angle movement
        self.angle += action
        self.direction[0] = math.sin(-math.radians(self.angle))
        self.direction[1] = -math.cos(math.radians(self.angle))

        # calculate the position from the direction and speed
        self.x += self.direction[0]*self.speed
        self.y += self.direction[1]*self.speed

class circleBoid(Boid):
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

        self.angle += 2
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

    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()

        leaderBoid = straightLineBoid(55, height / 2.0)
        # Define the start state for our rl algorithm
        learnerBoid = learnerBoid(35, heigh / 2.0, 90)

        # Move both boids
        leaderBoid.move()
        learnerBoid.move()
            
        screen.fill(white)

        # Draw the boids
        # Draw the leader
        boidRect = pygame.Rect(leadrect)
        boidRect.x = leaderBoid.x
        boidRect.y = leaderBoid.y
        screen.blit(lead, boidRect)

        # Draw the learner
        boidRect = pygame.Rect(birdrect)
        boidRect.x = learnerBoid.x
        boidRect.y = learnerBoid.y
        screen.blit(bird, boidRect)
        
        pygame.display.flip()
        pygame.time.delay(1)
   

# Perform |numTrials| of the following:
# On each trial, take the MDP |mdp| and an RLAlgorithm |rl| and simulates the
# RL algorithm according to the dynamics of the MDP.
# Each trial will run for at most |maxIterations|.
# Return the list of rewards that we get for each trial.
def simulate(rl, numTrials=10, maxIterations=1000, verbose=False,
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
        distance_old = util.distance(old_learner_loc, old_leader_loc)

        # Calculate new distance 
        new_learner_loc = newState[0]
        new_leader_loc = newState[1]
        distance_new = util.distance(new_learner_loc, new_leader_loc)

        reward = 0
        # Base reward on how the distance changes
        if distance_new < crashdistance:
            reward = -20
        elif distance_old > distance_new:
            reward = 5
        elif distance_old < distance_new:
            reward = -5

        return reward

    totalRewards = []  # The rewards we get on each trial
    for trial in range(numTrials):
        # We want to start doing the simulation
        # Let us start by placing down a the leader and
        # the learning follower
        leaderBoid = straightLineBoid(55, height / 2.0)
        # Define the start state for our rl algorithm
        learnerBoid = learnerBoid(35, heigh / 2.0, 90)

        # Define the start state that will be passed to our learning algorithm
        state = ((learnerBoid.x, learnerBoid.y), (leaderBoid.x, leaderBoid.y), leaderBoid.speed, (width, height))

        # We have to define the start state. We should start the bird close to the
        # follow bird
        sequence = [state]
        #totalDiscount = 1
        totalReward = 0
        for _ in range(maxIterations):
            # Get the action predicted by the bird learning algorithm
            action = rl.getAction(state)
            # Move the learning bird
            learnerBoid.move(action)

            # Move the leading bird
            leaderBoid.move()

            newState = ((learnerBoid.x, learnerBoid.y), (leaderBoid.x, leaderBoid.y), leaderBoid.speed, (width, height))
            
            reward = reward(state, newState)

            sequence.append(action)
            sequence.append(reward)
            sequence.append(newState)

            rl.incorporateFeedback(state, action, reward, newState)

            totalReward += totalDiscount * reward

            state = newState
        if verbose:
            print "Trial %d (totalReward = %s): %s" % (trial, totalReward, sequence)
        totalRewards.append(totalReward)
    return totalRewards


