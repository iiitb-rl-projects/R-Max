import numpy as np
import random
import sys
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
from gym import Env
from pylab import random, cos


class My_Environment():
    # () -> string
    # seps=0
    fileName = "DaPlot.csv"
    pos_x = 0.0
    pos_y = 0.0
    orient = 0.0
    walls = [[4, -10, 4, 12],
             [8, 30, 8, 8],
             [8, 3, 8, 5],
             [12, -10, 12, 16],
             [16, -10, 16, 6]]

    # This function checks for collisions with vertical walls if any and returns the point
    # of collision if so applies.

    def checkObsVertical(self, x1, y1, x2, y2, w1, z1, w2, z2):
        if (x1 == x2):
            return 0, [x2, y2]
        m1 = (y2 - y1) / (x2 - x1)
        c1 = y1 - x1 * (y2 - y1) / (x2 - x1)

        poi_x = w2
        poi_y = m1 * w2 + c1

        # print str(poi_x) + " " + str(poi_y)

        if abs(poi_x - x2) + abs(poi_x - x1) == abs(x2 - x1) and \
           abs(poi_y - y2) + abs(poi_y - y1) == abs(y2 - y1) and \
           abs(poi_y - z2) + abs(poi_y - z1) == abs(z2 - z1):
            return 1, [poi_x, poi_y]
        else:
            return 0, [x2, y2]

    # This Function checks for collisions with horizontal walls if any. As of now I'm
    # keeping this unused as I haven't programmed any horizontal walls into the matrix yet.

    def checkObsHorizontal(self, x1, y1, x2, y2, w1, z1, w2, z2):
        if (x1 == x2):

            poi_x = x1
            poi_y = z1

            # print str(poi_x) + " " + str(poi_y)

            '''print abs(poi_x - x2)
            print abs(poi_x - x1)
            print abs(x2 - x1)
            print abs(poi_y - y2)
            print abs(poi_y - y1)
            print abs(y2 - y1)
            print abs(poi_x - w1)
            print abs(poi_x - w2)
            print abs(w2 - w1)'''

            if abs(poi_x - x2) + abs(poi_x - x1) == abs(x2 - x1) and \
               abs(poi_y - y2) + abs(poi_y - y1) == abs(y2 - y1) and \
               abs(poi_x - w1) + abs(poi_x - w2) == abs(w2 - w1):
                return 1, [poi_x, poi_y]
            else:
                return 0, [x2, y2]

        if (y1 == y2): return 0, [x2, y2]

        m1 = (y2 - y1) / (x2 - x1)
        c1 = y1 - x1 * (y2 - y1) / (x2 - x1)

        poi_x = (z2 - c1) / m1
        poi_y = z2

        if abs(poi_x - x2) + abs(poi_x - x1) == abs(x2 - x1) and \
           abs(poi_y - y2) + abs(poi_y - y1) == abs(y2 - y1) and \
           abs(poi_x - w1) + abs(poi_x - w2) == abs(w2 - w1):
            return 1, [poi_x, poi_y]
        else:
            return 0, [x2, y2]

    def env_init(self):
        # thefile=open(self.fileName,"w")
        # thefile.close()
        return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1 OBSERVATIONS INTS " \
               "(0 499) ACTIONS INTS (0 2) REWARDS (-10.0 20.0) EXTRA TaxiEnv by Us."

    # () -> Observation
    def env_start(self):
        self.pos_x = 0.01 + random() * 3.8
        self.pos_y = 0.01 + random() * 3.8
        orient = np.pi / 2
        returnObs = Observation()
        returnObs.doubleArray = [self.pos_x, self.pos_y, orient]
        theFile = open(self.fileName, "a");

        theFile.write("%.2f, " % self.pos_x)
        theFile.write("\t");
        theFile.write("%.2f, " % self.pos_y)
        theFile.write("\n")

        theFile.close();
        return returnObs

    # (Action) -> Reward_observation_terminal
    def env_step(self, action):
        assert len(action.intArray) <= 2, "Expected 1 integer action."
        # assert action.intArray[0]>=0, "Expected action to be in [0,2]"
        # assert action.intArray[0]<3, "Expected action to be in [0,2]"
        theFile = open(self.fileName, "a");

        R = 0.0
        A = action.intArray[0]
        if not A in (0, 1, 2):
            print 'Invalid action:', A
            raise StandardError

        # If agent chooses to move forward.

        if (A == 0):

            # Potential new position is calculated using basic trigonometry and gaussian
            # noise is added to the result

            new_pos_x = self.pos_x + np.cos(self.orient) + np.random.normal(0.0, 0.1, 1)
            new_pos_y = self.pos_y + np.sin(self.orient) + np.random.normal(0.0, 0.1, 1)

            # Initializing variables that'd return the position of the agent.

            final_x = new_pos_x
            final_y = new_pos_y

            outOfBounds = 0

            # Checking if the new positions are within bounds. If not then appropriate new
            # postion is calculated and assigned.

            '''if(new_pos_x > 20):
                final_x = 19.95
                outOfBounds = 1
            if(new_pos_x < 0):
                final_x = 0.05
                outOfBounds = 1
            if(new_pos_y > 20):
                final_y = 19.95
                outOfBounds = 1
            if(new_pos_y < 0):
                final_y = 0.05
                outOfBounds = 1'''

            # If the new position is within bounds then check if there are any collisions on
            # the way

            if (outOfBounds == 0):
                for i in range(5):

                    x1 = self.pos_x
                    y1 = self.pos_y
                    x2 = new_pos_x
                    y2 = new_pos_y

                    w1 = self.walls[i][0]
                    z1 = self.walls[i][1]
                    w2 = self.walls[i][2]
                    z2 = self.walls[i][3]

                    # poi receives the x and y coordinates of the point of collision with wall
                    # if there is one.

                    status, poi = self.checkObsVertical(x1, y1, x2, y2, w1, z1, w2, z2)
                    if (status == 1): break

                # If there is a collision then loop with break at index i and status will be
                # set to 1. We can now use the coordinates of the wall and the point of
                # collision to calculate the appropriate new postion
                theFile.write("%.2f, " % self.pos_x)
                theFile.write("\t");
                theFile.write("%.2f, " % self.pos_y)
                theFile.write("\t");
                theFile.write("%.2f, " % new_pos_x)
                theFile.write("\t");
                theFile.write("%.2f, " % new_pos_y)
                theFile.write("\t");
                if (status == 1):
                    if (x1 < w1):
                        poi[0] = poi[0] - 0.05
                    if (x1 > w1):
                        poi[1] = poi[1] + 0.05

                final_x = poi[0]
                final_y = poi[1]

            if new_pos_x > 20:
                final_x = 19.95
                outOfBounds = 1
            if new_pos_x < 0:
                final_x = 0.05
                outOfBounds = 1
            if new_pos_y > 20:
                final_y = 19.95
                outOfBounds = 1
            if new_pos_y < 0:
                final_y = 0.05
                outOfBounds = 1

        else:
            if A == 1:
                self.orient = self.orient - np.pi / 6
                if self.orient < 0:
                    self.orient += 2 * np.pi
                final_x = self.pos_x
                final_y = self.pos_y
            else:
                if A == 2:
                    self.orient = self.orient + np.pi / 6
                    if self.orient > 2 * np.pi:
                        self.orient += -2 * np.pi
                    final_x = self.pos_x
                    final_y = self.pos_y

        returnRO = Reward_observation_terminal()

        self.pos_x = final_x
        self.pos_y = final_y

        if final_x > 16 and final_x < 20 and final_y > 0 and final_y < 4:
            returnRO.r = 10
            returnRO.terminal = True
        else:
            returnRO.r = -1
            returnRO.terminal = False

        returnRO.o = Observation()
        returnRO.o.doubleArray = [final_x, final_y, self.orient]

        theFile.write("%.2f, " % self.pos_x)
        theFile.write("\t");
        theFile.write("%.2f, " % self.pos_y)
        theFile.write("\n")

        theFile.close();
        return returnRO

    # () -> void
    def env_cleanup(self):
        pass

    # (string) -> string

    def env_message(self, inMessage):
        pass


if __name__ == "__main__":
    EnvironmentLoader.loadEnvironment(My_Environment())
