import random
import sys
import pickle
from rlglue.agent.Agent import Agent
from rlglue.agent import AgentLoader as AgentLoader
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.utils import TaskSpecVRLGLUE3
from random import Random
from My_Tiler import numTilings, tilecode, numTiles
from My_Tiler import numTiles
from pylab import *  # includes numpy
import random
import copy
from collections import defaultdict


class Rmax_agent(Agent):
    alpha = 0.05 / numTilings
    gamma = 1
    lmbda = 0.7
    Epi = Emu = epsilon = 0.1
    n = numTiles * 3  # 4 * 9 * 9 * 3
    F = [-1] * numTilings  # 4
    theta = np.zeros(n)
    e = np.zeros(n)
    rmax = 20
    C_t_a = defaultdict(lambda: defaultdict(lambda: 0))
    rsum = defaultdict(lambda: defaultdict(lambda: 0.0))
    rwrd = defaultdict(lambda: defaultdict(lambda: 0.0))
    m = 10
    policyFrozen = False
    exploringFrozen = False

    def egreedy(self, Qs, epsilon):
        # if rand() < epsilon:
            # return randint(3) # return a random action

        # else:
        return argmax(Qs)

    def Qs(self, F):
        # initialize Q[S, A, F]

        Q = np.zeros(3)

        # numActions
        # for every possible action a in F
        for a in xrange(3):
            # numTilings
            for i in F:
                # update Qa
                Q[a] = Q[a] + self.theta[i + (a * numTiles)]
        return Q

    def agent_init(self, taskSpecString):
        TaskSpec = TaskSpecVRLGLUE3.TaskSpecParser(taskSpecString)
        if TaskSpec.valid:
            self.numActions = TaskSpec.getIntActions()[0][1] + 1
        else:
            print "Task Spec could not be parsed: " + taskSpecString

        self.theta = -0.01 * rand(self.n)
        self.lastAction = Action()
        self.lastObservation = Observation()
        self.e = np.zeros(self.n)

    def agent_start(self, observation):
        S = observation.doubleArray
        A = 0
        # get a list of four tile indices
        tilecode(S[0], S[1], S[2], self.F)

        Q = self.Qs(self.F)

        # pick the action
        A = self.egreedy(Q, self.Emu)

        returnAction = Action()
        returnAction.intArray = [A]

        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)

        return returnAction

    def agent_step(self, reward, observation):
        Sprime = observation.doubleArray
        lastState = self.lastObservation.doubleArray
        lastAction = self.lastAction.intArray[0]
        R = reward
        # tile code takes state information, namely x,y and orientation and returns the tiles
        # which encode this state.
        tilecode(lastState[0], lastState[1], lastState[2], self.F)
        Q = self.Qs(self.F)

        # The agent maintains reward model for each (tile,action) pair, when you visit a state
        # and take action a and get reward 'R' this reward is distributed evenly among all the
        # tiles that encode that state.

        for i in self.F:
            # replacing traces
            self.e[i + (lastAction * numTiles)] = 1
            self.rsum[i][lastAction] += R
            self.C_t_a[i][lastAction] += 1
        # Until we have visited a (tile,action) pair M times we assume that it is unknown and
        # that the pair leads to maximum reward.
        # (tile,action)=(t,a) means that you take action a on tile t.
        # once a tile, action pair has been visited enough times it becomes known and we use
        # the average reward for that pair which is derived from the reward model
        # Then these reward values are used to derive a Q function which is used to get the
        # optimal policy/best action for that state.
        for i in self.F:
            if self.C_t_a[i][lastAction] >= self.m:
                self.rwrd[i][lastAction] = self.rsum[i][lastAction] / (self.C_t_a[i][lastAction] * 4)
            else:
                self.rwrd[i][lastAction] = self.rmax / 4

        R = 0
        for i in self.F:
            R += self.rwrd[i][lastAction]

        delta = R - Q[lastAction]

        tilecode(Sprime[0], Sprime[1], Sprime[2], self.F)

        Qprime = self.Qs(self.F)
        A = self.egreedy(Qprime, self.Emu)

        delta = delta + max(Qprime)

        # if not self.policyFrozen:
        # update theta
        self.theta = self.theta + self.alpha * delta * self.e

        # update e
        self.e = self.gamma * self.lmbda * self.e

        # update current state to next state for next iteration

        returnAction = Action()
        returnAction.intArray = [A]

        self.lastAction = copy.deepcopy(returnAction)
        self.lastObservation = copy.deepcopy(observation)

        return returnAction

    def agent_end(self, reward):
        lastState = self.lastObservation.doubleArray
        lastAction = self.lastAction.intArray[0]
        R = reward
        tilecode(lastState[0], lastState[1], lastState[2], self.F)
        Q = self.Qs(self.F)
        delta = R - Q[lastAction]

        for i in self.F:
            # replacing traces
            self.e[i + (lastAction * numTiles)] = 1

        if not self.policyFrozen:
            self.theta = self.theta + self.alpha * delta * self.e
            self.e = self.gamma * self.lmbda * self.e

    def agent_cleanup(self):
        pass

    def save_value_function(self, fileName):
        theFile = open(fileName, "w")
        pickle.dump(self.value_function, theFile)
        theFile.close()

    def load_value_function(self, fileName):
        theFile = open(fileName, "r")
        self.value_function = pickle.load(theFile)
        theFile.close()

    def agent_message(self, inMessage):

        #	Message Description
        # 'freeze learning'
        # Action: Set flag to stop updating policy
        #
        if inMessage.startswith("freeze learning"):
            self.policyFrozen = True
            return "message understood, policy frozen"

        # Message Description
        # unfreeze learning
        # Action: Set flag to resume updating policy
        #
        if inMessage.startswith("unfreeze learning"):
            self.policyFrozen = False
            return "message understood, policy unfrozen"

        # Message Description
        # freeze exploring
        # Action: Set flag to stop exploring (greedy actions only)
        #
        if inMessage.startswith("freeze exploring"):
            self.exploringFrozen = True
            return "message understood, exploring frozen"

        # Message Description
        # unfreeze exploring
        # Action: Set flag to resume exploring (e-greedy actions)
        #
        if inMessage.startswith("unfreeze exploring"):
            self.exploringFrozen = False
            return "message understood, exploring frozen"

        # Message Description
        # save_policy FILENAME
        # Action: Save current value function in binary format to
        # file called FILENAME
        #
        if inMessage.startswith("save_policy"):
            splitString = inMessage.split(" ")
            self.save_value_function(splitString[1])
            print "Saved."
            return "message understood, saving policy"

        # Message Description
        # load_policy FILENAME
        # Action: Load value function in binary format from
        # file called FILENAME
        #
        if inMessage.startswith("load_policy"):
            splitString = inMessage.split(" ")
            self.load_value_function(splitString[1])
            print "Loaded."
            return "message understood, loading policy"

        return "SampleSarsaAgent(Python) does not understand your message."


if __name__ == "__main__":
    AgentLoader.loadAgent(sarsa_agent())
