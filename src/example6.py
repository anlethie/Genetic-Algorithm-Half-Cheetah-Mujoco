import gym
from actor import GeneticNNActor
from genetics import evolve, render_from_file

HIDDEN_LAYERS = [10, 6]
SAVEFILE      = 'MsPacman_NN_10_6.txt'

try:
    env = gym.make('MsPacman-ram-v0')
    population = [GeneticNNActor(env.observation_space, env.action_space, hidden_layers=HIDDEN_LAYERS) for _ in range(100)]
    model = population[0]
    evolve(population, env,
        generations=1000, simulation_reps=5,
        p_mutation=0.05, mutation_scale=0.25,
        max_steps=100000, render_gens=None,
        savefile=SAVEFILE,
        savenum=3,
        allow_parallel=True
        )
except KeyboardInterrupt:
    print('Interrupted...')
    pass
finally:
    print('Top 3 Actors found:')
    render_from_file(SAVEFILE, model, env, num=3)