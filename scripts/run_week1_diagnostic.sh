#!/usr/bin/env bash
set -e
SOURCE=${1:-amazon_movies_and_tv}
TARGETS=${2:-amazon_cds_and_vinyl,steam}
EPOCHS=${3:-20}

python run_diagnostic.py --model semantic --source "$SOURCE" --targets "$TARGETS" --epochs "$EPOCHS"
python run_diagnostic.py --model dynamics --source "$SOURCE" --targets "$TARGETS" --epochs "$EPOCHS"
python run_diagnostic.py --model naive_fusion --source "$SOURCE" --targets "$TARGETS" --epochs "$EPOCHS"
python run_diagnostic.py --model gated_fusion --source "$SOURCE" --targets "$TARGETS" --epochs "$EPOCHS"
