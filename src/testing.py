from genetics import evolve, load_actors, dump_genomes, undump_genomes
from execution import simulate
import os

def test_actor_class(actor_class, env, population_size=100, savefile='test_actor_class.txt', tempfile='temp.txt', actor_args = {}, evolve_args = {}, render_args = {}):
    """Tests actors created by actor_class in env."""
    def new_actor():
        return actor_class(env.observation_space, env.action_space, **actor_args)

    def evolve_further():
        print('Loading population from',tempfile,'...')
        model = new_actor()
        population = []
        try:
            # population = load_actors(savefile, model, criterion='best', num=population_size)
            population = undump_genomes(tempfile, model)
            os.remove(tempfile)
        except:
            print('Could not load from',tempfile,'.')

        population = list(population) # enforce type of population
        # Add members to population as necessary
        population += [new_actor() for _ in range(population_size - len(population))]

        print('Evolving population...')
        population = evolve(population, env, savefile=savefile, dumpfile=tempfile, **evolve_args)
        dump_genomes(tempfile, population)


    def run_best(num = 1):
        print('Loading best from savefile...')
        model = new_actor()
        best_actors = None
        try:
            best_actors = load_actors(savefile, model, criterion='best', num=num)
        except:
            print('Could not load from',savefile,'. Exiting...')
            return

        for actor in best_actors:
            simulate(actor, env, render = True, **render_args)

    action = 'evolve'
    prompt = '[E]volve, [R]ender, or E[x]it? '
    while action != 'exit' and action != 'x':
        try:
            if action == 'evolve' or action == 'e':
                evolve_further()
            elif action == 'render' or action == 'r':
                run_best()
            # else pass
        except KeyboardInterrupt:
            pass
        action = input(prompt).lower()
        while action not in ['exit', 'x', 'evolve', 'e', 'render', 'r']:
            print('Did not recognize input', action)
            action = input(prompt).lower()