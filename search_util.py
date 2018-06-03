import heapq, collections, re, sys, time, os, random

############################################################
# Abstract interfaces for search problems and search algorithms.

class SearchProblem:
    # Return the start state.
    def startState(self): raise NotImplementedError("Override me")

    # Return whether |state| is an end state or not.
    def isEnd(self, state): raise NotImplementedError("Override me")

    # Return a list of (action, newState, cost) tuples corresponding to edges
    # coming out of |state|.
    def succAndCost(self, state): raise NotImplementedError("Override me")

class SearchAlgorithm:
    # First, call solve on the desired SearchProblem |problem|.
    # Then it should set two things:
    # - self.actions: list of actions that takes one from the start state to an end
    #                 state; if no action sequence exists, set it to None.
    # - self.totalCost: the sum of the costs along the path or None if no valid
    #                   action sequence exists.
    def solve(self, problem): raise NotImplementedError("Override me")

############################################################
# Uniform cost search algorithm (Dijkstra's algorithm).

class UniformCostSearch(SearchAlgorithm):
    def __init__(self, verbose=0):
        self.verbose = verbose

    def solve(self, problem):
        # If a path exists, set |actions| and |totalCost| accordingly.
        # Otherwise, leave them as None.
        self.actions = None
        self.totalCost = None
        self.numStatesExplored = 0

        # Initialize data structures
        frontier = PriorityQueue()  # Explored states are maintained by the frontier.
        backpointers = {}  # map state to (action, previous state)

        # Add the start state
        startState = problem.startState()
        frontier.update(startState, 0)

        while True:
            # Remove the state from the queue with the lowest pastCost
            # (priority).
            state, pastCost = frontier.removeMin()
            if state == None: break
            self.numStatesExplored += 1
            if self.verbose >= 2:
                print "Exploring %s with pastCost %s" % (state, pastCost)

            # Check if we've reached an end state; if so, extract solution.
            if problem.isEnd(state):
                self.actions = []
                while state != startState:
                    action, prevState = backpointers[state]
                    self.actions.append(action)
                    state = prevState
                self.actions.reverse()
                self.totalCost = pastCost
                if self.verbose >= 1:
                    print "numStatesExplored = %d" % self.numStatesExplored
                    print "totalCost = %s" % self.totalCost
                    print "actions = %s" % self.actions
                return

            # Expand from |state| to new successor states,
            # updating the frontier with each newState.
            for action, newState, cost in problem.succAndCost(state):
                if self.verbose >= 3:
                    print "  Action %s => %s with cost %s + %s" % (action, newState, pastCost, cost)
                if frontier.update(newState, pastCost + cost):
                    # Found better way to go to |newState|, update backpointer.
                    backpointers[newState] = (action, state)
        if self.verbose >= 1:
            print "No path found"

# Data structure for supporting uniform cost search.
class PriorityQueue:
    def  __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  # Map from state to priority

    # Insert |state| into the heap with priority |newPriority| if
    # |state| isn't in the heap or |newPriority| is smaller than the existing
    # priority.
    # Return whether the priority queue was updated.
    def update(self, state, newPriority):
        oldPriority = self.priorities.get(state)
        if oldPriority == None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    # Returns (state with minimum priority, priority)
    # or (None, None) if the priority queue is empty.
    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE: continue  # Outdated priority, skip
            self.priorities[state] = self.DONE
            return (state, priority)
        return (None, None) # Nothing left...


class MazeProblem(SearchProblem):
    def __init__(self, start, end, obstacles):
        self.start = start
        self.end = end
        self.obstacles = obstacles

    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        return self.start
        # END_YOUR_CODE

    def isEnd(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return state == self.end
        # END_YOUR_CODE

    def succAndCost(self, state):
        results = []
        pos_neg = [-1, 1]
        for delt in pos_neg:
            x_prime = state[0] + delt
            y_prime = state[1] + delt
            x = state[0]
            y = state[1]
            #check itself and its corners
            def allGood(x, y):
                poss = [0, 20]
                for x1 in poss:
                    for y1 in poss:
                        if (x + x1, y + y1) in self.obstacles:
                            if self.obstacles[(x + x1,y + y1)]:
                                return False
                            else: 
                                continue
                        else:
                            return False
                return True
            multiplier = 1

            # for i in range(-20, 40, 4):
            #     for j in range(-20, 40, 4):
            #         key = (x + i, y + j)
            #         if key in self.obstacles and self.obstacles[key]:
            #             multiplier = 10
            #             break

            if allGood(x_prime, y):
                results.append(((x_prime, y), (x_prime, y), multiplier))
            if allGood(x, y_prime):
                results.append(((x, y_prime), (x, y_prime), multiplier))
            if allGood(x_prime, y_prime):
                results.append(((x_prime, y_prime), (x_prime, y_prime), 1.5*multiplier))
            if allGood(x_prime, y - delt):
                results.append(((x_prime, y - delt), (x_prime, y - delt), 1.5*multiplier))
        return results   
        #END YOUR CODE