from hexapod.controllers.NEATController import Controller, tripod_gait, reshape, stationary
from hexapod.simulator import Simulator
import pymap_elites.map_elites_1.cvt as cvt_map_elites
import numpy as np
import neat
import pymap_elites.map_elites_1.common as cm
import pickle
import os

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

def evaluate_gait_parallel(x, duration=5):
    net = neat.nn.FeedForwardNetwork.create(x, config)
    # net.reset()
    leg_params = np.array(stationary).reshape(6, 5)
    # print(net.values)
    try:
        controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6, ann=net)
    except:
        return 0, np.zeros(6)
    simulator = Simulator(controller=controller, visualiser=False, collision_fatal=True)
    # contact_sequence = np.full((6, 0), False)
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            fitness = 0, np.zeros(6)
    # contact_sequence = np.append(contact_sequence, simulator.supporting_legs().reshape(-1, 1), axis=1)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    # summarise descriptor
    # descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    simulator.terminate()
    return fitness

filename = 'mapElitesOutput/NEAT/12_20000archive/archive_genome1196.pkl'
# filename = 'NEATOutput/bestGenomes/NEATGenome0.pkl'
with open(filename, 'rb') as f:
    genomes = pickle.load(f)

#print(len(genomes))
test = genomes
test = (list(test.values())[0])
print(test.best_genome())
print(evaluate_gait(test, duration = 5))

controller = Controller(stationary, body_height=0.15, velocity=0.5, crab_angle=-np.pi / 6, ann=winner_net,
                            printangles=True)
simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=[0])


while True:
    simulator.step()
