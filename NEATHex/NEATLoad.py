import neat
from hexapod.controllers.NEATController import Controller, tripod_gait, reshape
from hexapod.simulator import Simulator
import numpy as np
import multiprocessing
import os
import sys
import shutil
import pickle
import visualize as vz
from pureples.shared.visualize import draw_net

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

# with open("NEATStats55.pkl", 'rb') as f:
#     stats = pickle.load(f)
#     winner = stats.best_genome()
#     winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

with open(r"C:\Users\micha\PycharmProjects\neat-python\NEATOutput\bestGenomes\NEATGenome8.pkl", 'rb') as f:
    winner = pickle.load(f)
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

controller = Controller(tripod_gait, body_height=0.15, velocity=0.5, crab_angle=-np.pi / 6, ann=winner_net,
                            printangles=True)
simulator = Simulator(controller, follow=True, visualiser=True, collision_fatal=False, failed_legs=[0])


while True:
    simulator.step()