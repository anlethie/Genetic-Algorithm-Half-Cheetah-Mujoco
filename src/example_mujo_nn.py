import gym
from testing import test_actor_class
from actor import ModifiedGeneticNNActor as MGNNA
from genetics import top_selection

# 11 Observation
# 2 actuators
# The model may fall into local minimum, which accumulate alive points
# Our true goal is to gain the best running speed
# Modified mujoco-py walker2d.py to remove alive point


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    print("Observation space's shape: ", str(env.observation_space.low.shape))
    print("Action space's shape: ", str(env.action_space.low.shape))
    test_actor_class(MGNNA, env,
        savefile='HalfCheetah-v2-GNNAM.txt',
        population_size=200,
        actor_args={
                'hidden_layers': [6,6]
            },
        evolve_args={
                'generations': 1000,
                'simulation_reps': 1,
                'max_steps': 1000,
                'selection': lambda p: top_selection(p, cutoff=0.20),
                'keep_parents_alive' : True,
                'p_mutation': 0.2,
                'mutation_scale': 2.,
                'render_gens': 20,
                'savenum': 1,
                'allow_parallel':True
            },
        render_args={
                'fps': 60,
                'max_steps':3000
            }
        )