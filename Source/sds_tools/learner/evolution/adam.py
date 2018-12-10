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

df = pd.read_csv('C:\PycharmProjects\ml.services\Source\sds_tools\learner\evolution\\000b0000-ac12-0242-6daf-08d61ef809b0-result.csv')
df = df.iloc[:,3:259]
df = df.loc[:, (df != 0).any(axis=0)]
BOUND_UP = list(df.max())
BOUND_LOW = list(df.min())
BOUND_LOW.append(0)
df = df.iloc[0:200]
print(df)


df = df.to_json(orient='values',path_or_buf='starters.json')

smi_to_lat_model = load_model('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\ENUM2CAN\smi2lat.h5')
latent_to_states_model = load_model('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\ENUM2CAN\lat2state.h5')
sample_model = load_model('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\ENUM2CAN\samplemodel.h5')
charset = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\ENUM2CAN\charset.obj','rb'))
char_to_int = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\ENUM2CAN\char_to_int.obj','rb'))
smiles_len = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\ENUM2CAN\smiles_len.obj','rb'))
int_to_char = pickle.load(open('C:\PycharmProjects\ml.services\Source\sds_tools\learner\seq2seq\ENUM2CAN\int_to_char.obj','rb'))

given = '0.0,19.511613845825195,0.0,0.0,101.84548950195312,0.0,39.129398345947266,0.0,77.08615112304688,0.0,49.40758514404297,0.0,103.42547607421875,40.400203704833984,0.0,47.51445007324219,54.97690200805664,0.0,40.9547233581543,35.57423782348633,67.10903930664062,0.0,0.0,0.0,51.20613479614258,0.0,0.0,0.0,56.0975456237793,40.890567779541016,111.85730743408203,0.0,0.0,106.42203521728516,47.93569564819336,0.0,0.0,0.0,74.60598754882812,0.0,57.73796463012695,0.0,111.13580322265625,129.27513122558594,49.06441116333008,0.0,83.46475219726562,0.0,85.15855407714844,71.87627410888672,0.0,58.48630142211914,52.50952911376953,0.0,73.61285400390625,0.0,60.828861236572266,72.5362319946289,121.24211120605469,42.407691955566406,34.95346450805664,89.34648132324219,55.49089431762695,94.58905029296875,0.0,172.74020385742188,54.723663330078125,0.0,102.90151977539062,85.12236785888672,0.0,0.0,72.23360443115234,0.0,0.0,0.0,74.85816192626953,128.95040893554688,82.17964172363281,0.0,94.42498016357422,155.02664184570312,43.157535552978516,96.37812805175781,0.0,0.0,0.0,42.752899169921875,50.5090217590332,0.0,43.621829986572266,135.06605529785156,40.02004623413086,0.0,63.804588317871094,92.11248016357422,0.0,48.265132904052734,0.0,0.0,121.40558624267578,49.56248092651367,63.2352180480957,61.311546325683594,44.794803619384766,49.87528991699219,0.0,0.0,0.0,92.6469955444336,0.0,184.18992614746094,139.27108764648438,0.0,218.90431213378906,65.92842102050781,6.008143901824951,84.60381317138672,0.0,58.405704498291016,22.1331729888916,77.85977935791016,0.0,41.90808868408203,107.62860107421875,27.135223388671875,0.0,0.0,0.0,0.0,0.0,108.96160125732422,59.50330352783203,118.14600372314453,50.14487075805664,0.0,79.09473419189453,64.2027587890625,87.27764129638672,0.0,0.0,75.57288360595703,0.0,48.861812591552734,0.0,51.18389129638672,64.14624786376953,92.549560546875,61.64143753051758,97.0406265258789,0.0,0.0,0.0,0.0,65.12122344970703,0.0,53.5643310546875,76.93698120117188,0.0,44.362091064453125,116.67655944824219,38.66350555419922,0.0,0.0,0.0,0.0,0.0,100.56484985351562,38.19453430175781,0.0,70.59272766113281,58.61124038696289,0.0,0.0,23.79081916809082,0.0,58.61326599121094,154.3920440673828,56.0794792175293,31.37140464782715,115.35797119140625,64.31336212158203,55.418785095214844,0.0,38.841766357421875,113.07113647460938,0.0,0.0,0.0,53.31327438354492,70.05583953857422,110.50933074951172,0.0,0.0,0.0,52.083133697509766,0.0,0.0,29.5716609954834,36.2513542175293,0.0,36.090911865234375,0.0,101.65526580810547,99.85685729980469,0.0,66.85289764404297,61.308746337890625,0.0,0.0,59.77785110473633,94.66217803955078,89.78416442871094,0.0,89.25480651855469,75.43046569824219,61.54755401611328,101.5020523071289,0.0,45.96051788330078,97.4427261352539,0.0,0.0,57.75127410888672,84.43328857421875,39.7234001159668,0.0,68.55062103271484,98.57598876953125,103.91691589355469,45.874420166015625,31.459779739379883,40.12807846069336,49.90414047241211,0.0,0.0,0.0,25.711496353149414,121.99552917480469,0.0,51.547080993652344,75.97683715820312,72.50204467773438,33.95379638671875,0.0,118.1387939453125,51.91447448730469,0.0,0.0,0.0,0.0,120.96520233154297,77.3481674194336,0.0,0.0,91.91608428955078'
given = given.split(',')
given = [float(x) for x in given]
given_test = [x for x in given if x != 0.0]

non_zero_index = []
for i in range(len(given)):
    if given[i] != 0.0:
        non_zero_index.append(i)

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
            dissimilarity_to_benzene = (1 - TanimotoSimilarity(mol_fp,ref))
            print(dissimilarity_to_benzene)
            return bad_drug, dissimilarity_to_benzene
        except:
            return 9999,9999
    else:
        return 9999,9999


print(problem_drug_likeness(given_test))


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] ='\\usepackage{libertine}\n\\usepackage[utf8]{inputenc}'

seaborn.set(style='whitegrid')
seaborn.set_context('notebook')

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

def initIndividual(icls, content):
    return icls(content)

def initPopulation(pcls, ind_init, filename):
    with open(filename, "r") as pop_file:
        contents = json.load(pop_file)
    return pcls(ind_init(c) for c in contents)

toolbox = base.Toolbox()

NDIM = 151


toolbox.register("attr_float", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("individual_guess", initIndividual, creator.Individual)
toolbox.register("population_guess", initPopulation, list, toolbox.individual_guess, "starters.json")
toolbox.register("evaluate", problem_drug_likeness)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=5, indpb=0.05)
toolbox.register("select", tools.selNSGA2)

toolbox.pop_size = 50
mu = 50
toolbox.max_gen = 1000
toolbox.mut_prob = 0.3
toolbox.cross_prob = 0.7

def run_ea(toolbox, stats=None, verbose=False):
    pop = toolbox.population_guess()
    pop = toolbox.select(pop, len(pop))
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




