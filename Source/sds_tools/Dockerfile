# This Dockerfile builds base ML OSDR image
# "docker.your-company.com/osdr-sds-tools:latest"

# Base image
FROM "leanda/ml-base:latest"

LABEL maintainer="rick.zakharov@gmail.com"

ENV INSTALL_PATH /usr/app
RUN mkdir -p "$INSTALL_PATH" && chmod 777 "$INSTALL_PATH"
WORKDIR $INSTALL_PATH

# Add the app code
COPY . .

EXPOSE 5040
