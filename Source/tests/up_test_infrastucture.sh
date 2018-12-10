sudo docker network create --driver bridge osdr-sys-ml-test-net

cd /home/nick/ml.services/Source/tests
sudo docker-compose -f docker-compose.test-infrastructure.yml down

cd /home/nick/ml.services/Source/tests
sudo docker-compose -f docker-compose.test-infrastructure.yml up
