from hexapod.controllers.hyperNEATController import Controller, tripod_gait, reshape, stationary
from hexapod.simulator import Simulator
import pymap_elites.map_elites_1.cvt as cvt_map_elites
import numpy as np
import neat
import pymap_elites.map_elites_1.common as cm
import pickle
from pureples.hyperneat import create_phenotype_network
from pureples.shared import Substrate, run_hyper
from pureples.shared.visualize import draw_net


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

def evaluate_gait(x, duration=5):
    cppn = neat.nn.FeedForwardNetwork.create(x, CONFIG)
    # Create ANN from CPPN and Substrate
    net = create_phenotype_network(cppn, SUBSTRATE)
    # Reset net

    leg_params = np.array(stationary).reshape(6, 5)
    # Set up controller
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
                                ann=net, activations=ACTIVATIONS)
    except:
        return 0, np.zeros(6)
    # Initialise Simulator
    simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
    # Step in simulator
    contact_sequence = np.full((6, 0), False)
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            fitness = 0, np.zeros(6)
        contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0,
                               posinf=0.0, neginf=0.0)
    # Terminate Simulator
    simulator.terminate()
    # print(difference)
    # fitness = difference
    # Assign fitness to genome
    return fitness, descriptor

# def evaluate_gait(genome, config, duration=5):
#     net = neat.nn.FeedForwardNetwork.create(genome, config)
#     # net.reset()
#     leg_params = np.array(tripod_gait).reshape(6, 5)
#     # print(net.values)
#     try:
#         controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6, ann=net)
#     except:
#         return 0, np.zeros(6)
#     simulator = Simulator(controller=controller, visualiser=False, collision_fatal=False)
#     contact_sequence = np.full((6, 0), False)
#     for t in np.arange(0, duration, step=simulator.dt):
#         try:
#             simulator.step()
#         except RuntimeError as collision:
#             fitness = 0, np.zeros(6)
#         contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
#     fitness = simulator.base_pos()[0]  # distance travelled along x axis
#     # summarise descriptor
#     descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0,
#                                posinf=0.0, neginf=0.0)
#     simulator.terminate()
#     return fitness, descriptor


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'NEATHex/config-feedforward')


def makeGenomeCopy(genome, config, numCopies):
    genomeList = []
    for i in range(numCopies):
        x = genome.mutate(config.genome_config)
        genomeList.append(x)
    return genomeList


if __name__ == '__main__':
    with open("HyperNEATOutput/stats/hyperNEATStats8.pkl", 'rb') as f:
        stats = pickle.load(f)

    genomes = stats.best_genomes(10)
    winner = stats.best_genome()
    print(evaluate_gait(winner, 5))
    winner.mutate(config.genome_config)
    print(evaluate_gait(winner, 5))
    #print(winner)


    params = \
        {
            # more of this -> higher-quality CVT
            "cvt_samples": 1000000,
            # we evaluate in batches to parallelise
            "batch_size": 1000,
            # proportion of niches to be filled before starting (400)
            "random_init": 0.01,
            # batch for random initialization
            "random_init_batch": 2390,
            # when to write results (one generation = one batch)
            "dump_period": 5e6,
            # do we use several cores?
            "parallel": True,
            # do we cache the result of CVT and reuse?
            "cvt_use_cache": True,
            # min/max of parameters
            "min": 0,
            "max": 1,
        }


    archive = cvt_map_elites.compute(6, genomes, evaluate_gait, n_niches=200, max_evals=20000,
                                     log_file=open('log.dat', 'w'), params=params, variation_operator=cm.neatMutation)
