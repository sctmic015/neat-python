import warnings

import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import time

def average_fitness():
    start_time = time.time()
    dfMain = pd.read_csv(r'C:\Users\micha\PycharmProjects\neat-python\NEATOutput\genomeFitness\NEATFitnessHistory0.csv')
    dfMain.columns = ['Best NEAT', 'Average NEAT']
    for i in range(1, 20):
        filename = r'C:\Users\micha\PycharmProjects\neat-python\NEATOutput\genomeFitness\NEATFitnessHistory' + str(i) + '.csv'
        df = pd.read_csv(filename)
        df.columns = ['Best NEAT', 'Average NEAT']
        dfMain['Best NEAT'] = dfMain['Best NEAT'] + df['Best NEAT']
        dfMain['Average NEAT'] = dfMain['Average NEAT'] + df['Average NEAT']
    dfMain['Best NEAT'] = dfMain['Best NEAT'] / 20
    dfMain['Average NEAT'] = dfMain['Average NEAT'] /20
    dfMain.reset_index()

    dfMain2 = pd.read_csv(
        r'C:\Users\micha\PycharmProjects\neat-python\HyperNEATOutput\\genomeFitness\HyperNEATFitnessHistory0.csv')
    dfMain2.columns = ['Best HyperNEAT', 'Average HyperNEAT']
    for i in range(1, 20):
        filename = r'C:\Users\micha\PycharmProjects\neat-python\HyperNEATOutput\genomeFitness\HyperNEATFitnessHistory' + str(
            i) + '.csv'
        df = pd.read_csv(filename)
        df.columns = ['Best HyperNEAT', 'Average HyperNEAT']
        dfMain2['Best HyperNEAT'] = dfMain2['Best HyperNEAT'] + df['Best HyperNEAT']
        dfMain2['Average HyperNEAT'] = dfMain2['Average HyperNEAT'] + df['Average HyperNEAT']
    dfMain2['Best HyperNEAT'] = dfMain2['Best HyperNEAT'] / 20
    dfMain2['Average HyperNEAT'] = dfMain2['Average HyperNEAT'] / 20
    dfMain2.reset_index()

    dfMain = dfMain.append(dfMain2)
    lines = dfMain.plot.line()
    plt.xlabel("Number of Generations")
    plt.ylabel("Fitness")
    plt.show()
    print(time.time() - start_time)





average_fitness()
