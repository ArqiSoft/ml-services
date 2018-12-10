cd /home/nick/ml.services/Source/tests
sudo docker-compose -f docker-compose.test-ml-services.yml down

cd /home/nick/ml.services/Source/docker
sudo docker build --file Dockerfile_cpu -t "docker.your-company.com/osdr-ml-base:latest" .
# sudo docker build --file Dockerfile_gpu_cuda_base -t "docker.your-company.com/osdr-ml-cuda-base:latest" .
# sudo docker build --file Dockerfile_gpu -t "docker.your-company.com/osdr-ml-base:latest" .

cd /home/nick/ml.services/Source/sds_tools
sudo docker build -t "docker.your-company.com/sds-tools:latest" .

cd /home/nick/ml.services/Source/osdr_ml_modeler
sudo docker build --file Dockerfile_trainer -t "docker.your-company.com/osdr-ml-modeler-test:latest" .
sudo docker build --file Dockerfile_report_generator -t "docker.your-company.com/osdr-ml-training-reporter-test:latest" .
sudo docker build --file Dockerfile_optimizer -t "docker.your-company.com/osdr-ml-training-optimizer-test:latest" .

cd /home/nick/ml.services/Source/osdr_ml_predictor
sudo docker build --file Dockerfile_predictor -t "docker.your-company.com/osdr-ml-predictor-test:latest" .
sudo docker build --file Dockerfile_SSP -t "docker.your-company.com/osdr-ml-single-structure-predictor-test:latest" .

cd /home/nick/ml.services/Source/osdr_feature_vectors_calculator
sudo docker build -t "docker.your-company.com/osdr-feature-vectors-calculator-test:latest" .

cd /home/nick/ml.services/Source/tests
sudo docker build -t "docker.your-company.com/osdr-ml-test:latest" .

cd /home/nick/ml.services/Source/tests
sudo docker-compose -f docker-compose.test-ml-services.yml up
