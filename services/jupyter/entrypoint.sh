#!/bin/bash

eval $( fixuid )

mkdir -p /home/jupyter/.local/bin
mkdir -p /home/jupyter/.local/lib/python$(python --version | grep -Eo '[0-9]+[.][0-9]+')/site-packages

exec "$@"
