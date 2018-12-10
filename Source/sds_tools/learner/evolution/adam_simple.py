import array
import os
import pickle
import json
from os import listdir
import random
import numpy as np
import pandas as pd
from keras.models import load_model
from learner.evolution.sascorer import calculateScore
from learner.models import coeff_determination
from learner.seq2seq.sampler import latent_to_smiles
from rdkit import Chem
from rdkit.Avalon.pyAvalonTools import GetAvalonFP
from rdkit.Chem import Descriptors, AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity, DiceSimilarity
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn
from deap import algorithms, base, benchmarks, tools, creator
from eva import uniform, get_models

df = pd.read_csv('C:\PycharmProjects\ml.services\Source\sds_tools\learner\evolution\enum2can_sol.csv',)
df = df.iloc[:,3:259]
counter = 0
non_zero_index = []
non_zero_columns = []
flag = True
for column in df:
    flag = True
    for value in df[column].values:
        if value != 0.0:
            flag = False
    if flag is False:
        non_zero_index.append(counter)
        non_zero_columns.append(column)
    counter+=1
print(len(non_zero_index))


df = df[non_zero_columns]

BOUND_UP = list(df.max())
BOUND_LOW = list(df.min())
NDIM = len(non_zero_index)

stds = list(df.std(axis=0))
means = list(df.mean(axis=0))

df = df.iloc[200:226]
df.to_json(orient='values',path_or_buf='starters.json')


BOUND_UP = [x+0.0000000001+0.2*x for x in BOUND_UP]
BOUND_LOW = [x-0.3*x for x in BOUND_LOW]

smi_to_lat_model = load_model('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\enum2can\smi2lat.h5')
latent_to_states_model = load_model('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\enum2can\lat2state.h5')
sample_model = load_model('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\enum2can\samplemodel.h5')
charset = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\enum2can\charset.obj','rb'))
char_to_int = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\enum2can\char_to_int.obj','rb'))
smiles_len = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\enum2can\smiles_len.obj','rb'))
int_to_char = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\enum2can\int_to_char.obj','rb'))

print(len(non_zero_index))

def similarity(individual):

    final_vector = [0.0 for x in range(256)]
    individual_latent_vector = [x for x in individual]
    counter = 0
    for i in range(256):
        if i in non_zero_index:
            final_vector[i] = individual_latent_vector[counter]
            counter += 1

    final_vector = np.reshape(final_vector,(1, 256))
    smiles = latent_to_smiles(
        charset,smiles_len,char_to_int,int_to_char,
        latent_to_states_model,sample_model,final_vector,
        type='2_layers'
    )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule and smiles is not '' and len(smiles) != 1:
        try:
            mol_fp = GetAvalonFP(molecule, 2)
            ref = GetAvalonFP(Chem.MolFromSmiles('c1ccccc1'),2)
            dissimilarity_to_ref = (1 - TanimotoSimilarity(mol_fp,ref))
            print(Chem.MolToSmiles(molecule))
            print(dissimilarity_to_ref)
            return dissimilarity_to_ref,
        except:
            return 9999,
    else:
        return 9999,


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] ='\\usepackage{libertine}\n\\usepackage[utf8]{inputenc}'

seaborn.set(style='whitegrid')
seaborn.set_context('notebook')

def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)

########################################################################################################################

pop_size = 25
select_size = 10

CXPB, MUTPB = 0.5, 0.2

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, "starters.json")
toolbox.register("evaluate", similarity)
toolbox.register("mate", tools.cxUniform, indpb=1.0/(NDIM/20))
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=5.0, indpb=1.0/(NDIM/20))
toolbox.register("select", tools.selBest)


def valid(individual):

    if 9999 in individual.fitness.values:
        return False
    else:
        return True


def main():
    final_pop = []

    while len(final_pop) < pop_size:
        pop = toolbox.population(pop_size - len(final_pop))
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        pop = list(filter(valid, pop))
        final_pop.extend(pop)

    pop = final_pop

    fits = [ind.fitness.values[0] for ind in pop]
    # Variable keeping track of the number of generations

    g = 0
    # Begin the evolution
    while min(fits) != 0.0 and g < 1000:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        # Select the next generation individuals
        offspring = toolbox.select(pop, select_size)
        while len(offspring) < pop_size:
            pop = toolbox.population(pop_size - len(offspring))
            # Evaluate the entire population
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            pop = list(filter(valid, pop))
            offspring.extend(pop)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        final_offspring = []
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                counter = 0
                print("Crossover initialized")
                successfull = False
                while successfull is False:
                    print("Crossover try {}".format(str(counter)))
                    new_child1 = toolbox.clone(child1)
                    new_child2 = toolbox.clone(child2)
                    new_child1, new_child2 = toolbox.mate(new_child1, new_child2)
                    new_child1.fitness.values = toolbox.evaluate(new_child1)
                    new_child2.fitness.values = toolbox.evaluate(new_child2)
                    if 9999 not in new_child1.fitness.values and 9999 not in new_child2.fitness.values:
                        new_child1_fitness = new_child1.fitness.values[0]
                        new_child2_fitness = new_child2.fitness.values[0]
                        child1_fitness = child1.fitness.values[0]
                        child2_fitness = child2.fitness.values[0]
                        new_min_fit = min(new_child1_fitness, new_child2_fitness)
                        print('New min fir - {}'.format(str(new_min_fit)))
                        old_min_fit = min(child1_fitness, child2_fitness)
                        print('Old min fir - {}'.format(str(old_min_fit)))
                        if new_min_fit <= old_min_fit:
                            final_offspring.append(new_child1)
                            final_offspring.append(new_child2)
                            successfull = True
                    counter += 1
                    if counter == 5:
                        final_offspring.append(child1)
                        final_offspring.append(child2)
                        successfull = True
            else:
                final_offspring.append(child1)
                final_offspring.append(child2)

        fits_fo = [ind.fitness.values[0] for ind in final_offspring]

        for i in range(len(final_offspring)):
            if random.random() < MUTPB:
                print("Mutation initialized")
                max_counter = 0
                if final_offspring[i].fitness.values[0] == min(fits_fo):
                    available_tries = 20
                else:
                    available_tries = 5

                counter = 0
                successfull = False
                while successfull is False:
                    print("Mutation try {}".format(str(max_counter)))
                    new_mutant = toolbox.clone(final_offspring[i])
                    toolbox.mutate(new_mutant)
                    new_mutant.fitness.values = toolbox.evaluate(new_mutant)
                    if 9999 not in new_mutant.fitness.values and\
                            new_mutant.fitness.values[0] < final_offspring[i].fitness.values[0]:
                        final_offspring[i] = new_mutant
                        successfull = True
                        counter = 0
                    else:
                        max_counter += 1
                        if new_mutant.fitness.values[0] == final_offspring[i].fitness.values[0]:
                            counter += 1
                        if counter == available_tries - 1:
                            counter = 0
                            marker = False
                            while marker is False:
                                new_ind = toolbox.individual()
                                new_ind.fitness.values = toolbox.evaluate(new_ind)
                                if 9999 not in new_ind.fitness.values:
                                    final_offspring[i] = new_ind
                                    marker = True
                                    max_counter = 0
                                    successfull = True
                        if max_counter == available_tries:
                            break



        pop[:] = toolbox.select(final_offspring, select_size)

        fits = [ind.fitness.values[0] for ind in pop]

        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x * x for x in fits)
        std = abs(sum2 / length - mean ** 2) ** 0.5

        print(length)
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)

main()



