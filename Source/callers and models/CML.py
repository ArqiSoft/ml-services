import ntpath

from learner.classic_classifier import ClassicClassifier
from learner.dnn_classifier import DNNClassifier
from processor import sdf_to_csv

filepath = 'C:\PycharmProjects\ml.services\Data\\UBC_dataset_latest_class_U_not_1.sdf'
batch_size_dnn = 4
valuename = 'IC50_eGFP'
classname = 'cut_off_activity'

fptype = [{'Type': 'DESC'},{'Type': 'MACCS'},{'Type': 'ECFC','Size': 1024,'Radius':3},{'Type': 'AVALON','Size': 1024}]

test_set_size = 0.0
major_subsample = 1

layers = [64,64]

input_drop_out = 0.0
drop_out = 0.0
n_split = 10
optimizer='Nadam'
activation='selu'
l_rate=0.01
beta=0.0001
k_constraint = 4
mc_train_cut_off = 0.65


output_path = 'C:\\PycharmProjects\\ml.services\\Source\\callers and models'

dataframe = sdf_to_csv(filepath,fptype,value_name_list=valuename,cut_off=0.1)


# dataframe = pd.read_csv(filename)
# x = dataframe.values #returns a numpy array
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(x)
# headers = [x for x in range(797)]
# headers.append('Tox')
# dataframe = pd.DataFrame(x_scaled,columns=headers)
# print(dataframe)


classifier = DNNClassifier(ntpath.basename(filepath), classname, dataframe,test_set_size=test_set_size,
                                   major_subsample=major_subsample, fptype=fptype,n_split=n_split, output_path=output_path,
                                   scale="standard",n_iter_optimize=10)

dnn = classifier.train_model('extremegradientboostingregressor')
dnn.make_plots()
classifier.make_perfomance_csv()
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

classifier1 = ClassicClassifier(ntpath.basename(filepath), classname, dataframe,test_set_size=test_set_size,
                                   major_subsample=major_subsample, fptype=fptype,n_split=n_split, output_path=output_path,
                                   scale="standard")

RF = classifier1.train_model('randomforestclassifier')
RF.make_plots()
classifier1.make_perfomance_csv()

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




