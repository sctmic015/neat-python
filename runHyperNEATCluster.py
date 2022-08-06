import math
import pickle
import neat
import visualize
import neat.nn
import numpy as np
import multiprocessing
import os
import sys
import visualize as vz
import shutil

from hexapod.controllers.hyperNEATController import Controller, tripod_gait, reshape
from hexapod.simulator import Simulator
from pureples.hyperneat import create_phenotype_network
from pureples.shared import Substrate, run_hyper
from pureples.shared.visualize import draw_net

def evaluate_gait_parallel(genome, config, duration = 5):
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    # Create ANN from CPPN and Substrate
    net = create_phenotype_network(cppn, SUBSTRATE)
    # Reset net

    leg_params = np.array(tripod_gait).reshape(6, 5)
    # Set up controller
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
                                ann=net, activations=ACTIVATIONS)
    except:
        return 0, np.zeros(6)
    # Initialise Simulator
    simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
    # Step in simulator
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            fitness = 0, np.zeros(6)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    # Terminate Simulator
    simulator.terminate()
    # Assign fitness to genome
    return fitness



INPUT_COORDINATES = [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5),
                     (0.2, 0), (0.4, 0), (0.6, 0),
                     (0.2, -0.5), (0.4, -0.5), (0.6, -0.5),
                     (-0.6, -0.5), (-0.4, -0.5), (-0.2, -0.5),
                     (-0.6, 0), (-0.4, 0), (-0.2, 0),
                     (-0.6, 0.5), (-0.4, 0.5), (-0.2, 0.5),
                     (0, 0.25), (0, -0.25)]
OUTPUT_COORDINATES = [(0.2, 0.5), (0.4, 0.5), (0.6, 0.5),
                     (0.2, 0), (0.4, 0), (0.6, 0),
                     (0.2, -0.5), (0.4, -0.5), (0.6, -0.5),
                     (-0.6, -0.5), (-0.4, -0.5), (-0.2, -0.5),
                     (-0.6, 0), (-0.4, 0), (-0.2, 0),
                     (-0.6, 0.5), (-0.4, 0.5), (-0.2, 0.5)]
HIDDEN_COORDINATES = [[(0.2, 0.5), (0.4, 0.5), (0.6, 0.5),
                     (0.2, 0), (0.4, 0), (0.6, 0),
                     (0.2, -0.5), (0.4, -0.5), (0.6, -0.5),
                     (-0.6, -0.5), (-0.4, -0.5), (-0.2, -0.5),
                     (-0.6, 0), (-0.4, 0), (-0.2, 0),
                     (-0.6, 0.5), (-0.4, 0.5), (-0.2, 0.5)]]

# Pass configuration to substrate
SUBSTRATE = Substrate(
    INPUT_COORDINATES, OUTPUT_COORDINATES, HIDDEN_COORDINATES)
ACTIVATIONS = len(HIDDEN_COORDINATES) + 2

# Configure cppn using config file
CONFIG = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'NEATHex/config-cppn')

def run(gens):
    """
    Create the population and run the XOR task by providing eval_fitness as the fitness function.
    Returns the winning genome and the statistics of the run.
    """
    pop = neat.population.Population(CONFIG)
    stats = neat.statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.reporting.StdOutReporter(True))

    pe = neat.parallel.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_gait_parallel)
    winner = pop.run(pe.evaluate, gens)

    print("done")



    return winner, stats


if __name__ == '__main__':
    if os.path.exists("HyperNEATOutput") and os.path.isdir("HyperNEATOutput"):
        shutil.rmtree("HyperNEATOutput")
    numRuns = int(sys.argv[1])
    startIndex = int(sys.argv[2])
    endIndex = int(sys.argv[3])

    for i in range(startIndex, endIndex):
        WINNER, STATS = run(numRuns)  # Only relevant to look at the winner.
        print("This is the winner!!!")
        print(type(WINNER))
        print('\nBest genome:\n{!s}'.format(WINNER))
        STATS.save_genome_fitness(delimiter=',', filename='HyperNEATOutput/fitness_history' + i + '.csv')
        vz.plot_stats(STATS, ylog=False, view=True, filename='HyperNEATOutput/avg_fitness' + i + '.svg')
        vz.plot_species(STATS, view=True, filename='HyperNEATOutput/speciation' + i + '.svg')

        # CPPN for winner
        CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
        #with open("1000evals.pkl", 'rb') as f:
        #    CPPN = pickle.load(f)
        ## ANN for winner
        WINNER_NET = create_phenotype_network(CPPN, SUBSTRATE)
        outputName = "hyperneat" + i + ".pkl"

        with open('HyperNEATOutput/' + outputName, 'wb') as output:
            pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
        draw_net(CPPN, filename="HyperNEATOutput/hyperneatCPPN" + i)
        draw_net(WINNER_NET, filename="HyperNEATOutput/hyperneatWINNER" + i)
