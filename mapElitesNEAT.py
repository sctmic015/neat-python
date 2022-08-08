from hexapod.controllers.NEATController import Controller, tripod_gait
from hexapod.controllers.kinematic import Controller, reshape, tripod_gait
from hexapod.simulator import Simulator
import pymap_elites.map_elites_1.cvt as cvt_map_elites
import numpy as np
import neat
import pymap_elites.map_elites_1.common as cm

def evaluate_gait(x, duration=5):
    body_height, velocity, leg_params = reshape(x)
    try:
        controller = Controller(leg_params, body_height=body_height, velocity=velocity, period=1.0, crab_angle=-np.pi/6)
    except:
        return 0, np.zeros(6)
    simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
    contact_sequence = np.full((6, 0), False)
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            return 0, np.zeros(6)
        contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1,1), axis=1)
    fitness = simulator.base_pos()[0] # distance travelled along x axis
    # summarise descriptor
    descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    simulator.terminate()

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


if __name__ == '__main__':
    params = \
        {
            # more of this -> higher-quality CVT
            "cvt_samples": 1000000,
            # we evaluate in batches to parallelise
            "batch_size": 2390,
            # proportion of niches to be filled before starting (400)
            "random_init": 0.01,
            # batch for random initialization
            "random_init_batch": 2390,
            # when to write results (one generation = one batch)
            "dump_period": 5e6,
            # do we use several cores?
            "parallel": False,
            # do we cache the result of CVT and reuse?
            "cvt_use_cache": True,
            # min/max of parameters
            "min": 0,
            "max": 1,
        }

    archive = cvt_map_elites.compute(6, 32, evaluate_gait, n_niches=10, max_evals=10,
                                     log_file=open('log.dat', 'w'), params=params, variation_operator=cm.polynomial_mutation)
