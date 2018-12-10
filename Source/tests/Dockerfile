# This Dockerfile builds ML test container
# "docker.your-company.com/osdr-ml-test:latest"

# Base image
FROM "docker.your-company.com/sds-tools:latest"

MAINTAINER rick.zakharov@gmail.com

ENV INSTALL_PATH /usr/app
RUN mkdir -p "$INSTALL_PATH" && chmod 777 "$INSTALL_PATH"
WORKDIR $INSTALL_PATH

# Add the test code
COPY . .

EXPOSE 5090
