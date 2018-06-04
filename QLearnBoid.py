import math, random
from collections import defaultdict

# Performs Q-learning to learn to follow a leader for a boid.  
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearnBoid():
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)

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
        #return 1.0 / math.sqrt(self.numIters)
        return 0.1 * 1.0 / math.sqrt(self.numIters)

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
        
        #print self.featureExtractor(state, action)
        #print reward
        for f, v in self.featureExtractor(state, action):
            #if f == 'leader-delta':
                #print 'Weight: %f, update: %f, reward: %f' %(v, v * coefficient, reward)
            self.weights[f] -= (v * coefficient)

#caculates euclidean distance
def distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

def distanceObj(bird1, bird2):
    return math.sqrt((bird1.x - bird2.x)**2 + (bird1.y - bird2.y)**2)

def distanceBirdCoord(loc1, bird):
    return math.sqrt((loc1[0] - bird.x)**2 + (loc1[1] - bird.y)**2)

# Subtract vectors to get the direction vector
# loc1 is the old location, loc2 is the new location
def sub(loc2, loc1):
    return (loc2[0] - loc1[0], loc2[1] - loc1[1])


def threeBirdFlock(state, action):
    # State could be the 2 nearest birds?
    # Or the k nearest birds
    boid, leader, birds, velocity = state
    features = []

    # Distance to the leader
    dist_leader = distanceObj(boid, leader)

    # Move toward the general direction of the flock
    # We definitely will make this distance based
    # Get the average distance of the flock
    dist = dist_leader
    avg_x = leader.x
    avg_y = leader.y
    for bird in birds:
        avg_x += bird.x
        avg_y += bird.y
        dist += distanceObj(boid, bird)
    avg_dist = dist / float(len(birds) + 1)
    centroid = (avg_x / float(len(birds) + 1), avg_y / float(len(birds) + 1))
    dist_center = distance((boid.x, boid.y), centroid)

    boid_x, boid_y, boid_angle = boid.x, boid.y, boid.angle
    # Make our move
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

    # Distance to leader after move
    updated_distance = distanceBirdCoord((boid_x, boid_y), leader)

    # We still don't want to be too close to anyone!?
    # We could check if it is too close to any of the birds
    number_too_close = 0
    close_birds = []
    updated_total_dist = updated_distance
    if updated_distance < 30:
        close_birds.append(updated_distance)
    bird_distances = []
    for bird in birds:
        new_dist = distanceBirdCoord((boid_x, boid_y), bird)
        # See if we are too close
        if new_dist < 30:
            number_too_close += 1
            close_birds.append(new_dist)

        updated_total_dist += new_dist
        bird_distances.append(new_dist)
    updated_avg_dist = updated_total_dist / float(len(birds) + 1)

    # Make a feature to represent being too close to the first two
    close_birds.sort()
    # Closest too-close
    if len(close_birds) > 0:
        features.append(('closest', 1.0 / close_birds[0]))
    else:
        features.append(('closest', 0))

    if len(close_birds) > 1:
        features.append(('second', 1.0 / close_birds[0]))
    else:
        features.append(('second', 0))

    if len(close_birds) > 0:
        features.append(('num-close', number_too_close))
        return features


    updated_dist_center = distance((boid_x, boid_y), centroid)
    features.append(('centroid', updated_dist_center - dist_center))
    # Lets make a feature that says how many are too close

    # Let's make a feature to see if we got closer to the leader
    distance_delta = updated_distance - dist_leader
    #print 'dist %f' % (distance_delta)
    # We want to seek out leader if we are far
    #features.append(('leader-delta', distance_delta if dist_leader >= 50 else 0))
    features.append(('leader-delta', distance_delta))
    #features.append(('leader-delta', distance_delta))
    # Make a feature the distance to the "avg" flock
    #avg_delta = updated_avg_dist - avg_dist
    #features.append(('avg-dist', avg_delta))

    return features









def followLeaderBoidFeatureExtractorV2(state, action):
    # State features: Boid (x, y, angle), Pos of leader (x, y, angle), Set velocity
    boid, leader, velocity, size = state
    features = []

    boid_x, boid_y, boid_angle = boid
    leader_x, leader_y, leader_angle = leader

    old_distance = distance((boid_x, boid_y), leader)
    old_angle = boid_angle
    # Try not moving
    # The velocity thing does not work I don't think lol
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

    # Try updating the location of the leader
    #direction_lead_x = math.sin(math.radians(leader_angle))
    #direction_lead_y = -math.cos(math.radians(leader_angle))
    #leader_x += direction_lead_x * 3
    #leader_y += direction_lead_y * 3
    #updated_distance = distance((boid_x, boid_y), (leader_x, leader_y))
    

    distance_delta = updated_distance - old_distance
    # Saying if we are going to crash into the other bird (the number 20 can be changed)
    if updated_distance < 30:
        features.append(('too-close', 1 / updated_distance))
        features.append(('distance-delta', 0))
        features.append(('distance', 0))
    else:
        #features.append(('too-close', updated_distance if updated_distance < 20 else 0))
        features.append(('too-close', 0))
        features.append(('distance-delta', distance_delta))
        features.append(('distance', 1/updated_distance))
    # feauture to see simlarity of angels, compare the difference in angle before and after move
    angle_delta = math.fabs(math.fabs(old_angle - leader_angle) - math.fabs(boid_angle - leader_angle))
    angle_delta = math.fabs(boid_angle - leader_angle)
    angle_delta_val = 1.0 / angle_delta if angle_delta != 0 else 1
    #features.append(('angle-direction', angle_delta_val))

    return features


   



# # Think of other features
# def followTheLeaderBoidFeatureExtractor(state, action):
#     # State features
#     # Pos of boid
#     # Current direction of the boid
#     # Pos of leader
#     boid, leader, velocity, size = state
#     features = []

#     boid_x, boid_y, boid_angle = boid
#     leader_x, leader_y, leader_angle = leader

#     old_distance = distance((boid_x, boid_y), leader)
#     old_angle = boid_angle
#     # Try not moving
#     if action != None:
#         boid_angle += action
#         # Perform the action
#         direction_x = math.sin(math.radians(boid_angle))
#         direction_y = -math.cos(math.radians(boid_angle))


#         # calculate the position from the direction and speed
        
#         boid_x += direction_x * 3 
#         boid_y += direction_y * 3

#     updated_distance = distance((boid_x, boid_y), leader)
#     #print updated_distance

#     # print 'action: %f, distance: %f' %(action, updated_distance)

#     #features.append(('distance', updated_distance))

#     # Try inverse
#     #features.append(('distance-delta', updated_distance - old_distance))
#     distance_delta = updated_distance - old_distance
#     #distance_delta_val = 1.0 / distance_delta if distance_delta != 0 else 1.0 / - 0.5
#     features.append(('distance-delta', distance_delta))


#     #features.append(('inverse-distance', 1.0/updated_distance))

#     # Play with this
#     features.append(('too-close', 1.0 if updated_distance < 20 else 0))

#     # Can we make them go toward the same direction?
#     # Compare the difference in angle before and after move
#     angle_delta = math.fabs(math.fabs(old_angle - leader_angle) - math.fabs(boid_angle - leader_angle))
#     angle_delta = math.fabs(boid_angle - leader_angle)
#     angle_delta_val = 1.0 / angle_delta if angle_delta != 0 else 1
#     features.append(('angle-direction', angle_delta_val))

#     min_side_dist = float('inf')
#     for i in range(2):
#         dist = distance((boid_x, boid_y), (boid_x, i * size[1]))
#         if dist < min_side_dist:
#             min_side_dist = dist

#     for i in range(2):
#         dist = distance((boid_x, boid_y), (i * size[0], boid_y))
#         if dist < min_side_dist:
#             min_side_dist = dist

#     #features.append(('side-distance', 1.0/min_side_dist))

#     return features
    '''
    dx, dy = action

    # Calculate the new-coordinates of the boid
    # after taking the action
    boid_x += action_dist * math.sin(action_angle)
    boid_y += action_dist * math.cos(action_angle)

    # Calculate the new distance between the boid
    # and the leader
    updated_distance = distance((boid_x, boid_y), leader)

    features.append(("distance", updated_distance))
    
    # Add feature representing the new distance
    # between the boid and the leader
    features.append(("velocity", (round(velocity[0]), round(velocity[1]))))

    min_side_dist = float('inf')
    for i in range(2):
        dist = distance((boid_x, boid_y), (boid_x, i * size[1]))
        if dist < min_side_dist:
            min_side_dist = dist

    for i in range(2):
        dist = distance((boid_x, boid_y), (i * size[0], boid_y))
        if dist < min_side_dist:
            min_side_dist = dist

    features.append(("side_distance", min_side_dist))
    # Think of other features
    '''




