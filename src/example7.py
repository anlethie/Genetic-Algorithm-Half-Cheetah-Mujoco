import gym
from testing import test_actor_class
from actor import GeneticNNActor as GNNA

if __name__ == '__main__':
    env = gym.make('Pong-ram-v0')
    test_actor_class(GNNA, env,
        savefile='Pong_NN_4.txt',
        actor_args={
                'hidden_layers': [4]
            },
        evolve_args={
                'generations': 1000,
                'simulation_reps':5,
                'max_steps':10000,
                'p_mutation': 0.05,
                'render_gens': None,
                'savenum': 3,
            },
        render_args={
                'fps': 20,
                'max_steps':5000
            }
        )