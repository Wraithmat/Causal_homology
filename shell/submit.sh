#!/bin/bash

mkdir $1
cd $1

conda create --name devtools_scicomp python=3.9

conda activate devtools_scicomp

python -m pip install pytest

mkdir src/pyclassify scripts test shell experiments

touch src/pyclassify/__init__.py src/pyclassify/utils.py scripts/run.py shell/submit.sbatch shell/submit.sh experiments/config.yaml test/test_py

python -m pip freeze > requirements.txt

touch pyproject.toml

wget -O pyproject.toml https://raw.githubusercontent.com/dario-coscia/devtools_scicomp_project_2025/refs/heads/main/pyproject.toml

sed -i "9s/\"INSERT\"/$2/" pyproject.toml
sed -i "13s/\"INSERT\"/$3/" pyproject.toml
sed -i "9s/\"INSERT@gmail.com\"/$4/" pyproject.toml

echo -e "\n*.dat\n*.data" >> .gitignore
