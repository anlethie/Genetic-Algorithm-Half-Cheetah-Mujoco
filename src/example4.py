import gym
from actor import GeneticPerceptronActor
from execution import simulate
from genetics import evolve

env = gym.make('MountainCar-v0')
population = [GeneticPerceptronActor(env.observation_space, env.action_space) for _ in range(100)]
evolve(population, env, generations=101, simulation_reps=25, max_steps=30000, render_gens=10)
