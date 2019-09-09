import gym
from testing import test_actor_class
from actor import GeneticPerceptronActor as GPA

if __name__ == '__main__':
    env = gym.make('DuplicatedInput-v0')
    test_actor_class(GPA, env,
        savefile='DuplicatedInput_PA.txt',
        actor_args={
            },
        evolve_args={
                'generations': 1000,
                'simulation_reps':25,
                'max_steps':10000,
                'p_mutation': 0.03,
                'render_gens': None,
                'savenum': 3,
                'allow_parallel':False
            },
        render_args={
                'fps': 3,
                'max_steps':5000
            }
        )