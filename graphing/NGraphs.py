import warnings

import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import numpy as np


def average_fitness():
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
    lines = dfMain.plot.line()
    plt.show()



average_fitness()
