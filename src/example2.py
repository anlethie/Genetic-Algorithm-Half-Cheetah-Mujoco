import gym
import time
from actor import Actor
from actor import PerceptronActor
from actor import GeneticPerceptronActor
from execution import simulate

env = gym.make('CartPole-v1')
act = GeneticPerceptronActor(env.observation_space, env.action_space)

print('-- Without Rendering --')
print(time.time())
print(simulate(act, env, render=False))
print(time.time())
print('-- With Rendering --')
print(time.time())
print(simulate(act, env, render=True))
print(time.time())
input('Press ENTER to quit...')