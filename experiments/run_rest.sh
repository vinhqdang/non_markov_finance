#!/bin/zsh
export PYTHONPATH=../src:.
PY=/opt/miniconda3/envs/py313/bin/python
$PY -u exp04_evolution.py  > ../results/exp04.log 2>&1
$PY -u exp05_correlation.py > ../results/exp05.log 2>&1
$PY -u exp03_potential.py  > ../results/exp03.log 2>&1
echo "ALL DONE"
