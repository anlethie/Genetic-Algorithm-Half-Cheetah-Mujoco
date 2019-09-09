import gym
from testing import test_actor_class
from actor import ModifiedGeneticNNActor as MGNNA
from genetics import top_selection


if __name__ == '__main__':
    env = gym.make('HalfCheetah-v2')
    print("Observation space's shape: ", str(env.observation_space.low.shape))
    print("Action space's shape: ", str(env.action_space.low.shape))
    test_actor_class(MGNNA, env,
        savefile='Best-HalfCheetah-v2-GNNAM.txt',
        population_size=500, 
        actor_args={
                'hidden_layers': [3,3] 
            },
        evolve_args={
                'generations': 1001, 
                'simulation_reps': 1,
                'max_steps': 1000,
                'selection': lambda p: top_selection(p, cutoff=0.20),  
                'keep_parents_alive' : True, 
                'p_mutation': 0.2, 
                'mutation_scale': 0.5, 
                'render_gens': 20,
                'savenum': 1,
                'allow_parallel':True
            },
        render_args={
                'fps': 60,
                'max_steps':3000
            }
        )