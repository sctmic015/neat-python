import math
import pickle
import neat
import neat.nn
import numpy as np
import multiprocessing
import visualize as vz

from hexapod.controllers.hyperNEATController import Controller, tripod_gait, reshape
from hexapod.simulator import Simulator
from pureples.hyperneat import create_phenotype_network
from pureples.shared import Substrate, run_hyper, visualize
from pureples.shared.visualize import draw_net

def evaluate_gait_parallel(genome, config, duration = 5):
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    # Create ANN from CPPN and Substrate
    net = create_phenotype_network(cppn, SUBSTRATE)
    # Reset net

    leg_params = np.array(tripod_gait).reshape(6, 5)
    # Set up controller
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.9, period=1.0, crab_angle=-np.pi / 6,
                                ann=net, activations=ACTIVATIONS)
    except:
        return 0, np.zeros(6)
    # Initialise Simulator
    simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
    # Step in simulator
    difference = 0
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            difference += simulator.step()
        except RuntimeError as collision:
            fitness = 0, np.zeros(6)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    # Terminate Simulator
    simulator.terminate()
    # print(difference)
    #fitness = difference
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
                            'config-cppn')

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

    vz.plot_stats(stats, ylog=False, view=True)
    vz.plot_species(stats, view=True)

    return winner, stats


if __name__ == '__main__':
    WINNER, STATS = run(4)  # Only relevant to look at the winner.
    print("Ahoy there")
    print(STATS)
    print("This is the winner!!!")
    print(type(WINNER))
    print('\nBest genome:\n{!s}'.format(WINNER))

    # CPPN for winner
    CPPN = neat.nn.FeedForwardNetwork.create(WINNER, CONFIG)
    #with open("1000evals.pkl", 'rb') as f:
    #    CPPN = pickle.load(f)
    ## ANN for winner
    WINNER_NET = create_phenotype_network(CPPN, SUBSTRATE)

    # Create and run controller
    controller = Controller(tripod_gait, body_height=0.15, velocity=0.46, crab_angle=-1.57, ann=WINNER_NET, activations=ACTIVATIONS)
    simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=[0])

    with open('hyperneat_xor_cppn.pkl', 'wb') as output:
        pickle.dump(CPPN, output, pickle.HIGHEST_PROTOCOL)
    draw_net(CPPN, filename="hyperneat_xor_cppn")
    draw_net(WINNER_NET, filename="hyperneat_xor_winner")

    while True:
        simulator.step()