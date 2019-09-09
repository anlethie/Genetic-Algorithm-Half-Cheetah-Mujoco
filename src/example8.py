import gym
from testing import test_actor_class
from actor import GeneticPerceptronActor as GPA

if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    test_actor_class(GPA, env,
        savefile='MountainCar_PA_pop_100_pm_10.txt',
        population_size=100,
        actor_args={
            },
        evolve_args={
                'generations': 1000,
                'simulation_reps':5,
                'max_steps':10000,
                'p_mutation': 0.10,
                'render_gens': None,
                'savenum': 1,
            },
        render_args={
                'fps': 30,
                'max_steps':5000
            }
        )