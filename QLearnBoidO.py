import math, random
from collections import defaultdict


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

obstacles = defineMaze(12)

class QLearnBoidObstacles():
    def __init__(self, actions, discount, featureExtractor, obstacles, explorationProb=0.1):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.obstacles = obstacles

        # As a test lets set the weights
        #self.weights = {"distance": -1, "too-close": -1, 'distance-delta':-1, 'side-distance':1, 'inverse-distance':1}
        #, "side-distance": 0.2
        self.numIters = 0

    # Return the Q function associated with the weights and features

    def getQ(self, state, action):
        score = 0
        for f, f_value in self.featureExtractor(state, action):
            score += self.weights[f] * f_value
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # normalizes weights by scaling each weight by 100/sum of magnitude of the weights
    def normalizeWeights(self):
        score = 0
        for f in self.weights:
            score += abs(self.weights[f])

        for f in self.weights:
            self.weights[f] = 100*self.weights[f]/float(abs(score))

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    #helper function
    def printWeights(self):
        print self.weights

    # We will call this function with (s, a, r, s'), used to update |weights|.
    def incorporateFeedback(self, state, action, reward, newState):
        # w = w - n(Qopt(s,a) - (r + gamma(Vopt(s'))))phi(s, a)
        # Calc Vopt as max(Qopt(s', a') for all actions a')
        # Terminal state so no need to update
        v_opt = 0
        if newState != None:
            v_opt = max(self.getQ(newState, newAction) for newAction in self.actions(newState))
        
        coefficient = self.getStepSize() * (self.getQ(state, action) - reward - self.discount*v_opt)
        
        for f, v in self.featureExtractor(state, action):
            self.weights[f] -= (v * coefficient)

#caculates euclidean distance
def distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# Subtract vectors to get the direction vector
# loc1 is the old location, loc2 is the new location
def sub(loc2, loc1):
    return (loc2[0] - loc1[0], loc2[1] - loc1[1])


def allGood(x, y):
    poss = [0, 20]
    x = int(x)
    y = int(y)
    for x1 in poss:
        for y1 in poss:
            if (x + x1, y + y1) in obstacles:
                if obstacles[(x + x1,y + y1)]:
                    return 1
                else: 
                    continue
            else:
                return 1
    return 0


queue = []


def followLeaderBoidFeatureExtractorObstacles(state, action):
    # State features: Boid (x, y, angle), Pos of leader (x, y, angle), Set velocity
    boid, leader, velocity, size = state
    features = []

    while len(queue) <= 16:
        queue.append(leader)

    old_leader = queue.pop()

    boid_x, boid_y, boid_angle = boid
    leader_x, leader_y, leader_angle = leader

    old_distance = distance((boid_x, boid_y), leader)
    old_angle = boid_angle

    prev_x = boid_x
    prev_y = boid_y
    # Try not moving
    if action[0] != None:
        boid_angle += action[0]
        velocity += action[1]
        if velocity > 3:
            velocity = 3
        if velocity < 0:
            velocity = 0
        direction_x = math.sin(math.radians(boid_angle))
        direction_y = -math.cos(math.radians(boid_angle))

        # calculate the position from the direction and speed
        boid_x += direction_x * velocity
        boid_y += direction_y * velocity

    updated_distance = distance((boid_x, boid_y), leader)

    

    distance_delta = updated_distance - old_distance
    # Saying if we are going to crash into the other bird (the number 20 can be changed)
    if updated_distance < 60:
        features.append(('too-close', distance_delta))
        features.append(('distance-delta', 0))
        features.append(('distance', 0))
    else:
        #features.append(('too-close', updated_distance if updated_distance < 20 else 0))
        features.append(('too-close', 0))
        features.append(('distance-delta', distance_delta))
        features.append(('distance', 1/updated_distance))
    # feauture to see simlarity of angels, compare the difference in angle before and after move
    # angle_delta = math.fabs(math.fabs(old_angle - leader_angle) - math.fabs(boid_angle - leader_angle))
    # angle_delta = math.fabs(boid_angle - leader_angle)
    # angle_delta_val = 1.0 / angle_delta if angle_delta != 0 else 1
    #features.append(('angle-direction', angle_delta_val))


    # min_side_dist = float('inf')
    # for i in range(2):
    #     dist = distance((boid_x, boid_y), (boid_x, i * size[1]))
    #     if dist < min_side_dist:
    #         min_side_dist = dist

    # for i in range(2):
    #     dist = distance((boid_x, boid_y), (i * size[0], boid_y))
    #     if dist < min_side_dist:
    #         min_side_dist = dist

    # very_nearby = 0
    # bound_very = 25
    # for i in range(0,bound_very + 1):
    #     for j in range(0,bound_very + 1):
    #         key = (int(boid_x) + i, int(boid_y) + j)
    #         if key not in obstacles or obstacles[key] == True:
    #             very_nearby = 1
    #             break
    features.append(('very_nearby', allGood(boid_x, boid_y)))


    # there = 0
    # if (int(boid_x), int(boid_y)) not in obstacles or obstacles[key] == True:
    #     there = 1

    # features.append(('there', there))


    # count = 0
    # for i in range(-15, 35, 4):
    #     for j in range(-15, 35, 4):
    #         key = (int(boid_x) + i, int(boid_y) + j)
    #         if key not in obstacles or obstacles[key] == True:
    #             count += 1

    # features.append(('count', count))

    # features.append(('nearby', nearby))

    # if updated_distance > 100:
    #     old_distance = distance((prev_x, prev_y), old_leader)
    #     old_angle = boid_angle

    #     new_updated_distance = distance((boid_x, boid_y), old_leader)

    #     distance_delta = new_updated_distance - old_distance
    #     features.append(('distance-delta-old', distance_delta))
    # else:
    #     features.append(('distance-delta-old', 0))

    return features