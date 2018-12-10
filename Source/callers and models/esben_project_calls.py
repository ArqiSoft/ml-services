import ntpath
import pandas as pd
from learner.algorithms import ALGORITHM, TRAINER_CLASS, DNN_REGRESSOR, CODES
from learner.model_trainers import DNNRegressor, ClassicRegressor
from processor import sdf_to_csv
from rdkit import Chem
import sklearn

filepath = 'C:\PycharmProjects\ml-data-qsar\TEST\IGC50\IGC50_training.sdf'
filepath_test = 'C:\PycharmProjects\ml-data-qsar\TEST\IGC50\IGC50_prediction.sdf'
valuename = 'Tox'
classname = 'Soluble'

fptype = [{'Type': "ENUM2ENUM"},{'Type': "DESC"}]

test_set_size = 0
major_subsample = 1

layers = [64,64]

input_drop_out = 0.1
drop_out = 0.0
n_split = 10
optimizer='Nadam'
activation='relu'
l_rate=0.005
beta=0.00001
k_constraint = 4
mc_train_cut_off = 0.65

output_path = 'C:\\PycharmProjects\\ml.services\\Source\\callers and models'

dataframe = sdf_to_csv(filepath,fptype,value_name_list=valuename)
dataframe_test = sdf_to_csv(filepath_test,fptype,value_name_list=valuename)

regressor = ALGORITHM[TRAINER_CLASS][DNN_REGRESSOR](
    ntpath.basename(filepath), valuename, dataframe,test_set_size=test_set_size,
    fptype=fptype,n_split=n_split, output_path=output_path,
    scale="standard",manual_test_set=dataframe_test)

dnn = regressor.train_model(CODES[DNN_REGRESSOR])
dnn.make_plots()
regressor.make_perfomance_csv()

# dataframe = pd.read_csv(filename)
# x = dataframe.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# headers = [x for x in range(797)]
# headers.append('Tox')
# dataframe = pd.DataFrame(x_scaled,columns=headers)
# print(dataframe)

# classifier = ClassicClassifier(ntpath.basename(filepath), classname, dataframe,test_set_size=test_set_size,
#                                    major_subsample=major_subsample, fptype=fptype,n_split=n_split, output_path=output_path,
#                                    scale="standard")
#
# NB = classifier.train_model('naivebayes')
# NB.make_plots()
# classifier.make_perfomance_csv()
#
#
# classifier = ClassicClassifier(ntpath.basename(filepath), classname, dataframe,test_set_size=test_set_size,
#                                    major_subsample=major_subsample, fptype=fptype,n_split=n_split, output_path=output_path,
#                                    scale="standard")
#
# DT = classifier.train_model('decisiontree')
# DT.make_plots()
# classifier.make_perfomance_csv()
#
# classifier = ClassicClassifier(ntpath.basename(filepath), classname, dataframe,test_set_size=test_set_size,
#                                    major_subsample=major_subsample, fptype=fptype,n_split=n_split, output_path=output_path,
#                                    scale="standard")
#
# RF = classifier.train_model('randomforestclassifier')
# RF.make_plots()
# classifier.make_perfomance_csv()

# classifier = ClassicClassifier(ntpath.basename(filepath), classname, dataframe,test_set_size=test_set_size,
#                                    major_subsample=major_subsample, fptype=fptype,n_split=n_split, output_path=output_path,
#                                    scale="standard")
# LR = classifier.train_model('linearregression')
# LR.make_plots()
# classifier.make_perfomance_csv()
#
# classifier = ClassicClassifier(ntpath.basename(filepath), classname, dataframe,test_set_size=test_set_size,
#                                    major_subsample=major_subsample, fptype=fptype,n_split=n_split, output_path=output_path,
#                                    scale="standard")
#
# SVM = classifier.train_model('supportvectormachineclassifier')
# SVM.make_plots()
# classifier.make_perfomance_csv()
# classifier.train_model(2)
# classifier.make_plots()


# train_dnn_valid(classifier,layers,batch_size_dnn=batch_size_dnn,k_fold=k_fold,
#                 drop_out=drop_out,input_drop_out=input_drop_out,optimizer=optimizer,
#                 activation=activation, l_rate=l_rate, beta=beta)


