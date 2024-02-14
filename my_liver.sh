#!/bin/bash
module load python/3.6
module load scikit-learn/0.20.3

liver_data="liver_data.csv"

preprocess_script="preprocess.py"
train_model_script="train_model.py"
evaluate_model_script="evaluate_model.py"

python $preprocess_script $liver_data

train_model_output=$(python $train_model_script $liver_data)
# shellcheck disable=SC2086
my_liver=$(echo $train_model_output | grep -oP 'KNeighborsClassifier\(n_neighbors=\K\d+')

echo "Model: $my_liver"

python $evaluate_model_script $liver_data "$my_liver"