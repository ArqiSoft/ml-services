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
from rdkit.Chem import Descriptors, AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import seaborn
from deap import algorithms, base, benchmarks, tools, creator
from eva import uniform, get_models

df = pd.read_csv('C:\PycharmProjects\ml.services\Source\sds_tools\learner\evolution\can2enum_sol.csv',)
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

stds = list(df.std(axis=0))
means = list(df.mean(axis=0))

df = df.iloc[200:301]
df.to_json(orient='values',path_or_buf='starters.json')


BOUND_UP = [x+0.0000000001 for x in BOUND_UP]
BOUND_LOW = [x for x in BOUND_LOW]

smi_to_lat_model = load_model('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\CAN2ENUM\smi2lat.h5')
latent_to_states_model = load_model('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\CAN2ENUM\lat2state.h5')
sample_model = load_model('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\CAN2ENUM\samplemodel.h5')
charset = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\CAN2ENUM\charset.obj','rb'))
char_to_int = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\CAN2ENUM\char_to_int.obj','rb'))
smiles_len = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\CAN2ENUM\smiles_len.obj','rb'))
int_to_char = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\CAN2ENUM\int_to_char.obj','rb'))

print(len(non_zero_index))

def problem_drug_likeness(individual):

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
    if molecule:
        try:
            logP = Descriptors.MolLogP(molecule)
            logP_score = (1.575 - logP)**2
            SA_score = calculateScore(molecule)
            print(Chem.MolToSmiles(molecule))
            bad_drug = logP_score + SA_score
            mol_fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2)
            ref = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles('c1ccccc1'),2)
            dissimilarity_to_ref = (1 - TanimotoSimilarity(mol_fp,ref))
            print((bad_drug,dissimilarity_to_ref))
            return bad_drug, dissimilarity_to_ref
        except:
            return 9999,9999
    else:
        return 9999,9999


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] ='\\usepackage{libertine}\n\\usepackage[utf8]{inputenc}'

seaborn.set(style='whitegrid')
seaborn.set_context('notebook')


creator.create("FitnessMin", base.Fitness, weights=(-0.001,-1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)

toolbox = base.Toolbox()

NDIM = len(non_zero_index)


toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, "starters.json")
toolbox.register("evaluate", problem_drug_likeness)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=10)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=10, indpb=0.01)
toolbox.register("select", tools.selNSGA2)

toolbox.pop_size = 250
mu = 100
toolbox.max_gen = 1000
toolbox.mut_prob = 0.5
toolbox.cross_prob = 0.5

def valid(individual):

    if 9999 in individual.fitness.values:
        return False
    else:
        return True


def run_ea(toolbox, stats=None, verbose=False):
    final_pop = []

    while len(final_pop) < toolbox.pop_size:
        pop = toolbox.population(toolbox.pop_size - len(final_pop))
        # Evaluate the entire population
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        pop = list(filter(valid, pop))
        final_pop.extend(pop)
        print('#############################################################' + str(len(final_pop)))

    pop = toolbox.select(final_pop, len(final_pop))
    return algorithms.eaMuPlusLambda(pop, toolbox, mu=mu,
                                     lambda_=toolbox.pop_size,
                                     cxpb=toolbox.cross_prob,
                                     mutpb=toolbox.mut_prob,
                                     stats=stats,
                                     ngen=toolbox.max_gen,
                                     verbose=verbose)

res,_ = run_ea(toolbox)
fronts = tools.emo.sortLogNondominated(res, len(res))
df = pd.DataFrame(fronts)
df.to_csv('results.csv')



