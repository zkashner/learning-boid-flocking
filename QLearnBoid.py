import util, math, random
from collections import defaultdict
from util import ValueIteration

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
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
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

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        # w = w - n(Qopt(s,a) - (r + gamma(Vopt(s'))))phi(s, a)
        # Calc Vopt as max(Qopt(s', a') for all actions a')
        # Terminal state so no need to update
        #if newState == None:
            #Vopt = 0 
        Vopt = 0
        if newState != None:
            new_actions = self.actions(newState)
            Vopt = max(self.getQ(newState, a_new) for a_new in new_actions)

        #n(Qopt(s,a) - (r + gamma(Vopt(s'))))
        scalar_value = self.getStepSize() * (self.getQ(state, action) - (reward + self.discount * Vopt))
        # Update w
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = self.weights[f] - scalar_value * v
        # END_YOUR_CODE


def distance(loc1, loc2):
    return math.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

# Subtract vectors to get the direction vector
# loc1 is the old location, loc2 is the new location
def sub(loc2, loc1):
    return (loc2[0] - loc1[0], loc2[1] - loc1[1])

def followTheLeaderBoidFeatureExtractor(state, action):
    # State features
    # Pos of boid
    # Current direction of the boid
    # Pos of leader
    boid, velocity, leader = state
    features = []

    boid_x, boid_y = boid
    leader_x, leader_y = leader

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
    # detween the boid and the leader


    

    # Think of other features






