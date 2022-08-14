import neat
from hexapod.controllers.NEATController import Controller, tripod_gait, reshape, stationary
from hexapod.simulator import Simulator
import numpy as np
import multiprocessing
import os
import sys
import shutil
import pickle
import visualize as vz
from pureples.shared.visualize import draw_net


## x feedforward neuralnet
def evaluate_gait(genomes, config, duration=5):
    for genome_id, genome in genomes:
        genome.fitness = 4.0
        net = neat.nn.RecurrentNetwork.create(genome, config)
        leg_params = np.array(stationary).reshape(6, 5)
        # print(net.values)
        try:
            controller = Controller(leg_params, body_height=0.15, velocity=0.5, period=1.0, crab_angle=-np.pi / 6,
                                    ann=net)
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
        genome.fitness = fitness


def evaluate_gait_parallel(genome, config, duration=5):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
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


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'NEATHex/config-feedforward')

def runNeat(gens):
    p = neat.Population(config)
    stats = neat.statistics.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.StdOutReporter(True))

    pe = neat.parallel.ParallelEvaluator(multiprocessing.cpu_count(), evaluate_gait_parallel)
    winner = p.run(pe.evaluate, gens)
    return winner, stats

if __name__ == '__main__':
    if not os.path.exists("NEATOutput"):
        os.mkdir("NEATOutput")
        if not os.path.exists("NEATOutput/genomeFitness"):
            os.mkdir("NEATOutput/genomeFitness")
        if not os.path.exists("NEATOutput/graphs"):
            os.mkdir("NEATOutput/graphs")
        if not os.path.exists("NEATOutput/bestGenomes"):
            os.mkdir("NEATOutput/bestGenomes")
        if not os.path.exists("NEATOutput/stats"):
            os.mkdir("NEATOutput/stats")
    numRuns = int(sys.argv[1])
    startIndex = int(sys.argv[2])
    endIndex = int(sys.argv[3])
    for i in range(startIndex, endIndex+1):
        winner, stats = runNeat(numRuns)

        print("This is the winner!!!")
        print('\nBest genome:\n{!s}'.format(winner))
        i = str(i)
        stats.save_genome_fitness(delimiter=',', filename='NEATOutput/genomeFitness/NEATFitnessHistory' + i + '.csv')
        vz.plot_stats(stats, ylog=False, view=True, filename='NEATOutput/graphs/NEATAverageFitness' + i + '.svg')
        vz.plot_species(stats, view=True, filename='NEATOutput/graphs/NEATSpeciation' + i + '.svg')

        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

        outputNameGenome = "NEATGenome" + i + ".pkl"
        outputNamePopulation = "NEATStats" + i + ".pkl"

        with open('NEATOutput/bestGenomes/' + outputNameGenome, 'wb') as output:
            pickle.dump(winner, output, pickle.HIGHEST_PROTOCOL)
        with open('NEATOutput/stats/' + outputNamePopulation, 'wb') as output:
            pickle.dump(stats, output, pickle.HIGHEST_PROTOCOL)
        draw_net(winner_net, filename="NEATOutput/graphs/NEATWINNER" + i)



