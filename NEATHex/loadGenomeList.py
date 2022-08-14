import math
import pickle
import neat
import neat.nn
import numpy as np
import multiprocessing

from hexapod.controllers.hyperNEATController import Controller, tripod_gait, reshape
from hexapod.simulator import Simulator
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
                            'config-cppn')

with open("archive_genome11.pkl", 'rb') as f:
     genomes = pickle.load(f)
     print(type(genomes))
     winner = genomes[5]
     CPPN = neat.nn.FeedForwardNetwork.create(winner, CONFIG)
     WINNER_NET = create_phenotype_network(CPPN, SUBSTRATE)

# with open("hyperneat6.pkl", 'rb') as f:
#     CPPN = pickle.load(f)
#     WINNER_NET = create_phenotype_network(CPPN, SUBSTRATE)

# Create and run controller
controller = Controller(tripod_gait, body_height=0.15, velocity=0.5, crab_angle=-1.57, ann=WINNER_NET, activations=ACTIVATIONS)
simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=True, failed_legs=[])


while True:
    simulator.step()