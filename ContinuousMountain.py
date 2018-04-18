import random
import sys
from rlglue.environment.Environment import Environment
from rlglue.environment import EnvironmentLoader as EnvironmentLoader
from rlglue.types import Observation
from rlglue.types import Action
from rlglue.types import Reward_observation_terminal
from gym import Env
from pylab import random, cos

class My_Environment(Env):
	# () -> string
	#seps=0
	position = 0.0
	velocity=0.0
	def env_init(self):
		return "VERSION RL-Glue-3.0 PROBLEMTYPE episodic DISCOUNTFACTOR 1 OBSERVATIONS INTS (0 499) ACTIONS INTS (0 2) REWARDS (-10.0 20.0) EXTRA TaxiEnv by Us."
	
	# () -> Observation
 	def env_start(self):
		self.position = -0.6 + random()*0.2
		returnObs=Observation()
		returnObs.doubleArray=[self.position,0.0]
		return returnObs
	
	# (Action) -> Reward_observation_terminal
	def env_step(self,action):
		
		assert len(action.intArray)<=2,"Expected 1 integer action."
		#assert action.intArray[0]>=0, "Expected action to be in [0,2]"
		#assert action.intArray[0]<3, "Expected action to be in [0,2]"
		A=action.intArray[0]
		if not A in (0, 1, 2):
			print 'Invalid action:', A
			raise StandardError
			
		R = -1 if A == 1 else -1.5
		A = A - 1
		
		self.velocity += 0.001*A - 0.0025*cos(3*self.position)

		if self.velocity < -0.07:
			self.velocity = -0.07
		elif self.velocity >= 0.07:
			self.velocity = 0.06999999

		self.position += self.velocity

		if self.position < -1.2:
			self.position = -1.2
			self.velocity = 0.0

		returnRO=Reward_observation_terminal()
		returnRO.r=R*1.0
		returnRO.o=Observation()
		returnRO.o.doubleArray=[self.position,self.velocity]
		if self.position>=0.5:
			returnRO.terminal=True
			returnRO.r=20.0
		else:
			returnRO.terminal=False	
		#self.seps=self.seps+1
		#if self.seps >50:
		#	returnRO.terminal=TRUE
		return returnRO
	
	# () -> void
	def env_cleanup(self):
		pass
	
	# (string) -> string
	
	def env_message(self,inMessage):
		pass

	

if __name__=="__main__":
	EnvironmentLoader.loadEnvironment(My_Environment())
