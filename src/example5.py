if __name__ == '__main__':
    import gym
    from actor import GeneticPerceptronActor
    from execution import simulate
    from genetics import evolve

    env = gym.make('CartPole-v1')
    population = [GeneticPerceptronActor(env.observation_space, env.action_space) for _ in range(100)]
    evolve(population, env,
            generations=21, simulation_reps=25,
            max_steps=20000, render_gens=1,
            savefile='CartPole_Actors.txt',
            savenum=1
            )
