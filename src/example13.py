import gym
from testing import test_actor_class
from genetics import top_selection
from actor import GeneticPerceptronActor as GPA

if __name__ == '__main__':
    env = gym.make('Copy-v0')
    test_actor_class(GPA, env,
        savefile='Copy_PA.txt',
        actor_args={
            },
        evolve_args={
                'generations': 1000,
                'simulation_reps':25,
                'max_steps':10000,
                'p_mutation': 0.03,
                'selection': lambda p: top_selection(p, cutoff=0.40),
                'render_gens': None,
                'savenum': 3,
                'allow_parallel':False
            },
        render_args={
                'fps': 3,
                'max_steps':5000
            }
        )