#!/usr/bin/env bash

# Get parent directory of the directory of the current file
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DIR="$(dirname "$DIR")"

meeplemate-cli build-retriever --type hypothetical_queries --input_file munchkin_rules/ --tgi_endpoint http://tgi:80 --output_file munchkin_rules_hypothetical_queries_retriever.pkl
meeplemate-cli build-retriever --type parent_document --input_file munchkin_rules/ --tgi_endpoint http://tgi:80 --output_file munchkin_rules_parent_document_retriever.pkl
