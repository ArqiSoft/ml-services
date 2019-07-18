# Base image
FROM leanda/ml-tools:latest

LABEL maintainer="rick.zakharov@gmail.com"

ENV INSTALL_PATH /usr/app
RUN mkdir -p "$INSTALL_PATH" && chmod 777 "$INSTALL_PATH"
WORKDIR $INSTALL_PATH

# Add the app code
COPY . .

EXPOSE 5011
