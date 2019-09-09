import numpy as np
import random as rnd

from execution import simulate

# Had some trouble getting parallel computations to work.
# If there are issues, disable this flag
ALLOW_PARALLEL = True

try:
    assert ALLOW_PARALLEL,'Parallel computation manually disabled.'
    import gym
    from actor import GeneticPerceptronActor
    from joblib import Parallel, delayed
    # Actually execute a simple test of the parallel system, to make sure it works
    N_JOBS = 8
    # PARALLEL_TEST = Parallel(n_jobs=N_JOBS)(delayed(lambda x: x**0.5)(i) for i in range(100))
    env = gym.make('CartPole-v1')
    PARALLEL_TEST = Parallel(n_jobs=N_JOBS)(delayed(lambda a: simulate(a, env))(a)
        for a in [GeneticPerceptronActor(env.observation_space, env.action_space) for _ in range(100)])
    USE_PARALLEL  = True
    print('Parallel computation enabled.')
except Exception as e:
    USE_PARALLEL = False
    print('Warning! Parallel computation disabled.')
    print(e)

try:
    from multiprocessing import cpu_count
    N_JOBS = cpu_count()
    print('Detected', N_JOBS, 'CPUs.')
except Exception as e:
    print('Warning! Could not detect number of CPUs. Assuming', N_JOBS, '.')

# A "Genome" is just a Numpy array that's defined to have floating-point data between 0 and 1
# Standardizing what a genome is for our purposes lets us manipulate them in the abstract, 
#   so these functions can be applied to any problem environment.

def mutate(genome, p=1.0, scale=0.25):
    """Takes a genome and randomly mutates it.
The probability of mutating each element is p.
The mutation is drawn from a normal distribution with std. dev. = scale."""
    new_genome = genome.copy()
    for i in range(genome.size):
        if np.random.rand() < p:
            # new_genome[i] = np.random.rand()
            new_genome[i] = min(1., max(0., np.random.normal(new_genome[i], scale)))
    return new_genome

def crossover(genome1, genome2):
    """Takes two genomes and performs a cross-over operation at a random point."""
    assert genome1.size == genome2.size
    i = np.random.randint(1, genome1.size-1) # Don't allow 0-length cross-overs
    new_genome1 = genome1.copy()
    new_genome2 = genome2.copy()
    new_genome1[i:] = genome2[i:].copy()
    new_genome2[:i] = genome1[:i].copy()
    return new_genome1,new_genome2

def roulette_selection(scored_genomes):
    flo = min(s for s,_ in scored_genomes)
    fhi = max(s for s,_ in scored_genomes)

    """Returns a mating pool from the scored_genomes according to a roulette-sampling style selection algorithm."""
    # Perform the scaling: this shifts all numbers to be strictly positive
    scored_genomes = [( (score - flo) + 0.1 * abs(flo) , genome ) for score,genome in scored_genomes]
    total_score = sum(s for s,_ in scored_genomes)
    # Re-write the scores as cumulative scores:
    cscored_genomes = [scored_genomes[0]]
    for score,genome in scored_genomes[1:]:
        cscored_genomes.append((cscored_genomes[-1][0] + score, genome))

    def sample():
        r = np.random.rand() * total_score
        # Find the first cumulative score that exceeds the random number
        return next(genome for cscore,genome in cscored_genomes if cscore >= r)

    N = len(scored_genomes)
    # perform roulette-wheel sampling:
    return [sample() for _ in range(N)]

def top_selection(scored_genomes, cutoff=0.5):
    """Returns genomes.in the top 'cutoff' fraction of scores."""
    N = len(scored_genomes)
    cutoff_index = int(np.floor(N * cutoff))
    survivors    = [g for _,g in sorted(scored_genomes, key=lambda sg:-sg[0])[:cutoff_index]]
    return survivors

def run_generation(
        population, environment,
        p_mutation=0.01,
        mutation_scale=0.25,
        selection=roulette_selection,
        keep_parents_alive=False,
        simulation_reps=100,
        max_steps=1000,
        savefile=None,
        savenum=1,
        allow_parallel=True,
        max_jobs=None
        ):
    """Executes one full generation, starting with the given population.
Returns the new population.
p_mutation - the chance that a particular value in an offspring genome is changed
simulation_reps - the number of times to execute each actor in the environment, to account for random variation in initial environmental conditions
max_steps - the maximum number of simulation steps for each run
"""
    # TODO: write this function. Sketch:
    # x simulate each actor in population against environment some number of times, average scores
    #   Use those averaged scores to generate next population's genomes, copied from previous population according to roulette wheel or exponential distribution selection
    #     Alternatively, take the selection strategy as a functional parameter, to be tuned later
    #   Use mutate and cross-over on the resulting next generation
    #   Build GeneticActors from those new genomes

    def run_actor(actor):
        """Returns the actor's average score and genome."""
        score = np.mean([
            simulate(actor, environment, max_steps=max_steps, render=False)
            for _ in range(simulation_reps)
        ])
        return (score,actor.get_genome())

    population = list(population) # demand that this is a list, not an iterable or something weird.
    if USE_PARALLEL and allow_parallel:
        # When available, use parallel evaluation to speed up the evaluation of the population
        scored_genomes = Parallel(n_jobs=N_JOBS if max_jobs == None else min(N_JOBS, max_jobs))(
            delayed(run_actor)(actor) for actor in population)
    else:
        # Otherwise, not a big deal, just do things sequentially
        scored_genomes = [run_actor(actor) for actor in population]

    # flo,worst_genome = min(scored_genomes)
    # fhi,best_genome  = max(scored_genomes)
    flo,worst_genome = scored_genomes[0]
    fhi,best_genome  = scored_genomes[0]
    for s,g in scored_genomes[1:]:
        if s < flo:
            flo = s
            worst_genome = g
        if s > fhi:
            fhi = s
            best_genome = g
    if savefile != None:
        with open(savefile, 'a') as f:
            for j,(s,g) in enumerate(sorted(scored_genomes, key=lambda x:-x[0])):
                if j >= savenum:
                    break
                print('{s},{g}'.format(s=s, g=list(g)), file=f)
    
    mating_pool = selection(scored_genomes)

    def gen_children():
        parents = rnd.sample(mating_pool, 2)
        return crossover(*parents)

    parents_genomes = []
    offspring_genomes = []
    num_children = len(scored_genomes)

    if keep_parents_alive:
        parents_genomes = mating_pool
        num_children = len(scored_genomes) - len(parents_genomes)
    
    # Perform cross-over N//2 times, since each produces 2 offspring
    # Simultaneously execute mutations on each child
    offspring_genomes = [mutate(x, p=p_mutation, scale=mutation_scale) for _ in range(num_children // 2) for x in gen_children()]
    
    # merge both
    next_generation = parents_genomes + offspring_genomes

    # For debugging
    # print("Parents: " + str(len(parents_genomes)))
    # print("Offsprings: " + str(len(offspring_genomes)))
    # print("Next generation: " + str(len(next_generation)))
    

    # need some actor from the original pool, just to generate the new actors
    actor = population[0]

    return [actor.from_genome(genome) for genome in next_generation],actor.from_genome(best_genome),actor.from_genome(worst_genome)


def evolve(
        initial_population,
        environment,
        generations=100,
        p_mutation=0.01,
        mutation_scale=0.25,
        selection=roulette_selection,
        keep_parents_alive=False,
        simulation_reps=100,
        max_steps=1000,
        render_gens=10,
        savefile=None,
        dumpfile=None,
        savenum=1,
        allow_parallel=True,
        max_jobs=None
        ):
    """Runs selection and simulation on initial_population for specified number of generations.
Returns the final generation.
Renders a random individual from the population every render_gens generations.
p_mutation - the chance that a particular value in an offspring genome is changed
simulation_reps - the number of times to execute each actor in the environment, to account for random variation in initial environmental conditions
max_steps - the maximum number of simulation steps for each run
"""
    population = initial_population
    try:
        for i in range(generations):
            population,best_actor,worst_actor = run_generation(
                    population, environment,
                    p_mutation=p_mutation,
                    mutation_scale=mutation_scale,
                    selection=selection,
                    simulation_reps=simulation_reps,
                    keep_parents_alive=keep_parents_alive,
                    max_steps=max_steps,
                    savefile=savefile,
                    savenum=savenum,
                    allow_parallel=allow_parallel,
                    max_jobs=max_jobs
                )

            if render_gens != None and (i % render_gens) == 0:
                print('---=== Generation', i, '===---')
                simulate(best_actor, environment, video_postfix=i, render=True, max_steps=max_steps)
    except Exception as e:
        print(e) # For debugging
        if dumpfile != None:
            dump_genomes(dumpfile, population)
    return population

def dump_genomes(dumpfile, population):
    genomes = [a.get_genome() for a in population]
    with open(dumpfile, 'wb') as f:
        np.save(f, genomes)

def undump_genomes(dumpfile, model_actor):
    with open(dumpfile, 'rb') as f:
        genomes = np.load(f)
    return [model_actor.from_genome(g) for g in genomes]

def load_genomes(savefile, criterion='best', num=1):
    """Loads genomes (and their scores) from a savefile. Returns a list of genomes.
criterion can be 'best', 'worst', or 'first'
"""
    saved = []
    with open(savefile, 'r') as f:
        for line in f:
            score,genome = line.split(',', 1)
            score  = float(score)
            genome = np.array(eval(genome))
            if len(saved) < num:
                saved.append((score,genome))
            elif criterion == 'first':
                break
            elif criterion == 'best' and saved[0][0] < score:
                saved[0] = (score,genome)
            elif criterion == 'worst' and saved[-1][0] > score:
                saved[-1] = (score,genome)
            else:
                continue
            saved.sort(key=lambda x: x[0])
    return saved

def load_actors(savefile, model_actor, criterion='best', num=1):
    scored_genomes = load_genomes(savefile, criterion, num)
    return [model_actor.from_genome(g) for s,g in scored_genomes]

def render_from_file(savefile, model_actor, env, criterion='best', num=1, max_steps=1000):
    scored_genomes = load_genomes(savefile, criterion, num)
    for s,g in scored_genomes:
        actor = model_actor.from_genome(g)
        print('Score:', s)
        simulate(actor, env, render=True, max_steps=max_steps)