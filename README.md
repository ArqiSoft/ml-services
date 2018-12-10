# OSDR Machine Learning #

Collection of tools, libraries and services for Machine Learning.

# Environment Variables #

```
OSDR_BLOB_SERVICE_TOKEN_URL                              https://identity.your-company.com/core/connect/token
OSDR_BLOB_SERVICE_URL                                    https://api.dev.dataledger.io/blob/v1/api/blobs
OSDR_LOG_FOLDER                                          C:\Projects\SDS\_Logs
OSDR_ML_MODELER_CLIENT_SECRET                            umbrella
OSDR_RABBIT_MQ                                           amqp://guest:guest@localhost:5672/%2F
OSDR_RABBIT_MQ_HOST                                      localhost
OSDR_RABBIT_MQ_LOGIN                                     guest
OSDR_RABBIT_MQ_ML_EXCHANGE_MODEL_TRAINING_PROGRESS_EVENT "Sds.MachineLearning.Integration.Domain.Events.ModelTrainingProgress"
OSDR_RABBIT_MQ_ML_EXCHANGE_MODEL_TRAINING_FAILED_EVENT   "Sds.MachineLearning.Integration.Domain.Events.ModelTrainingFailed"
OSDR_RABBIT_MQ_ML_EXCHANGE_MODEL_TRAINED_EVENT           "Sds.MachineLearning.Integration.Domain.Events.ModelTrained"
OSDR_RABBIT_MQ_ML_EXCHANGE_MODEL_TRAINING_ABORTED_EVENT  "Sds.MachineLearning.Integration.Domain.Events.ModelTrainingAborted"
OSDR_RABBIT_MQ_ML_EXCHANGE_MODEL_REPORT_EVENT            "Sds.MachineLearning.Integration.Domain.Events.ModelTrainingReportCreated"
OSDR_RABBIT_MQ_ML_QUEUE_CREATE_MODEL_COMMAND             "Sds.Osdr.Domain.BackEnd - Sds.Osdr.Domain.BackEnd.Sagas.MachineLearningModelCreationSaga - Sds.MachineLearning.Integration.Domain.Commands.TrainModel"
OSDR_RABBIT_MQ_ML_QUEUE_ABORT_MODEL_TRAINING_COMMAND     "Sds.Osdr.Domain.BackEnd - Sds.Osdr.Domain.BackEnd.Sagas.MachineLearningModelCreationSaga - Sds.MachineLearning.Integration.Domain.Commands.AbortModelTraining"
OSDR_RABBIT_MQ_PASS                                      guest
OSDR_RABBIT_MQ_PORT                                      5672
```

## Running in CLI ##

Install packages:

```
pip install -r requirements.txt
conda install -c rdkit rdkit=2017.03.3
```

Command (`x` stands for parameter value):

```
./Start_ML.sh --help --class-name x --folder x -n x --radius x --bins x --major-subsamle x --test-set-size x --n-split x --cut-off x --<model>
```

Arguments:

```
--help (-h) - show this
--class-name (-l) - class name (default Active)
--folder (-f) - /full/path to directory with .sdf's (default current)
--output-path (-q) - /full/path to output directory ( /output dir with data and reports will be created there)
-n - number or cores (default 4)
--radius (-r) - fingerprint radius (default 3)
--bins (-i) - number of bins (default 1024)
--major-subsample (-m) - major_subsample (default 0.2)
--test-set-size (-t) - test set size (default 0.2)
--n-split (-s) - n split (default 4)
--cut-off (-o) - cut-off
```

Methods (default all):

```
--naive-bayes (-a) - Naive Bayes
--linear-regression (-b) - Linear Regression
--decision-tree (-c) - Decision Tree
--random-forest (-d) - Random Forest
--support-vector-machine (-e) - Support Vector Machine
--dnn (-d) - Deep Neural Network
```

Output goes to "Full_report.html"

## Running in Docker ##

### Create and run OSDR ML Docker ###

Build base OSDR docker image (if any of the packages changed):

```
cd ./docker
docker build -t "docker.your-company.com/osdr-ml-base:latest" .
cd ..
```

Build Modeler docker image:

```
cd ./osdr_ml_modeler
docker build -t "docker.your-company.com/osdr-ml-modeler:latest" .
cd ..
```

Build Predictor docker image:

```
cd ./osdr_ml_predictor
docker build -t "docker.your-company.com/osdr-ml-predictor:latest" .
cd ..
```

Run OSDR+ML docker-compose

Go to OSDR repository

```
cd ./Source/Services/.docker
docker-compose -f docker-compose.osdr.services.local.yml up
```

If you need to debug anything inside the Docker, attach to an image and launch `bash`:

```
docker rm osdr-ml
docker run --name osdr-ml-modeler -it docker.your-company.com/osdr-ml-modeler bash
docker run --name osdr-ml-predictor -it docker.your-company.com/osdr-ml-predictor bash
```

### Debug scripts using `miniconda3` ###

Mount your local drive to get accesses to all files:

```
docker run --name miniconda3 -v c:/Users:/notebooks/data -it continuumio/miniconda3
```

Install packages and their dependencies

```
pip install -r requirements.txt
```

Install RDKit and its dependencies

```
conda install -c rdkit rdkit=2017.03.3
```

Test if ML script can be run in this docker:

```
cd Source
cli_ml.py Estrogenreceptoralpha-Homosapiens-SP-B-CHEMBL206-Binding-K.sdf full_report.html
```

`sds_tools` entire folder (for simplicity)

```
python .Source/cli_ml.py -i "./Data/Estrogenreceptoralpha-Homosapiens-SP-B-CHEMBL206-Binding-K.sdf" -c any -v Value --cut-off 1500 -p Relation -r 3 -b 1024 -m .5 --naive-bayes
```

Results will be saved in `.Source/classic_ML_Estrogenreceptoralpha-Homosapiens-SP-B-CHEMBL206-Binding-K_ECFP_3_1024_cut_1500_subsample_0.5` folder

## Tests and coverage ##
### Start tests ###
1) Activate ml.services environment
2) Start ml.services/Source/tests/start_services_tests.py script
### Tests coverage ###
1) Activate ml.services environment
2) Install coverage module
```pip install coverage```
3) Start ml.services/Source/tests/check_coverage.sh script

That script will create folder named htmlcov. Finde Index.html in that folder and open it on browser to look at tests coverage.

## Resources ##

* [Conda](https://conda.io/docs/)
* [Docker](https://www.docker.com/)


