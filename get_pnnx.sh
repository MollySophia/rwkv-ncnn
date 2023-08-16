#!/bin/sh

__version="20230816"

wget https://github.com/pnnx/pnnx/releases/download/${__version}/pnnx-${__version}-ubuntu.zip \
    && unzip pnnx-${__version}-ubuntu.zip && mv pnnx-${__version}-ubuntu/pnnx . \
    && rm -r pnnx-${__version}-ubuntu pnnx-${__version}-ubuntu.zip