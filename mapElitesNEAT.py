from hexapod.controllers.NEATController import Controller, tripod_gait, reshape, stationary
from hexapod.simulator import Simulator
import pymap_elites.map_elites_1.cvt as cvt_map_elites
import numpy as np
import neat
import pymap_elites.map_elites_1.common as cm
import pickle

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'NEATHex/config-feedforward')

def evaluate_gait(x, duration=5):
    net = neat.nn.FeedForwardNetwork.create(x, config)
    # Reset net

    leg_params = np.array(stationary).reshape(6, 5)
    # Set up controller
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
                                ann=net)
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

def load_genomes():
    genomes = []
    for i in range(20):
        filename = 'NEATOutput/stats/NEATStats' + str(i) + '.pkl'
        with open(filename, 'rb') as f:
            stats = pickle.load(f)
            tempGenome = stats.best_unique_genomes(10)
            genomes += tempGenome
    print(len(genomes))
    return genomes

if __name__ == '__main__':
    genomes = load_genomes()
    params = \
        {
            # more of this -> higher-quality CVT
            "cvt_samples": 1000000,
            # we evaluate in batches to parallelise
            "batch_size": 50,
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

    archive = cvt_map_elites.compute(6, genomes, evaluate_gait, n_niches=500, max_evals=500,
                                     log_file=open('mapElitesOutput/NEAT/log1.dat', 'w'), archive_file='mapElitesOutput/NEAT/archiveTest', params=params,
                                     variation_operator=cm.neatMutation)