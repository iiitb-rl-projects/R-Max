import random
import sys
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
from gym.envs.toy_text import taxi


# My_Environment is inheriting the class TaxiEnv from gym/envs/toy_text/taxi.py

class My_Environment(taxi.TaxiEnv):
    # () -> string
    def env_init(self):
        # initializes the taxi environment map and states and rewards and initial state distribution.
        taxi.TaxiEnv.__init__(self);
        return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1 OBSERVATIONS INTS " \
               "(0 499) ACTIONS INTS (0 5) REWARDS (-10.0 20.0) EXTRA TaxiEnv by Us."

    # () -> Observation
    def env_start(self):
        # seed and reset are defined in gym/envs/toy_text/discrete.py which is included in taxi.py
        # chooses a fresh start state.
        self.seed()
        self.reset()
        returnObs = Observation()
        returnObs.intArray = [self.s]
        return returnObs

    # (Action) -> Reward_observation_terminal
    def env_step(self, action):
        # the taxi can take 6 actions, east,west,north,south,pickup and dropoff.
        assert len(action.intArray) == 1, "Expected 1 integer action."
        assert action.intArray[0] >= 0, "Expected action to be in [0,5]"
        assert action.intArray[0] < 6, "Expected action to be in [0,5]"
        # step function is defined in gym/envs/toy_text/discrete.py, the DiscreteEnv class defined
        # in there is inherited by TaxiEnv which MY_Environment inherits.
        s1, r1, d1, k1 = self.step(action.intArray[0])
        returnRO = Reward_observation_terminal()
        returnRO.r = r1 * 1.0
        returnRO.o = Observation()
        returnRO.o.intArray = [s1]
        returnRO.terminal = d1
        return returnRO

    # () -> void
    def env_cleanup(self):
        pass

    # (string) -> string

    def env_message(self, inMessage):
        pass


if __name__ == "__main__":
    EnvironmentLoader.loadEnvironment(My_Environment())
