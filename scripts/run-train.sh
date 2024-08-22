#!/bin/bash

# Copyright 2024 OKHADIR Hamza
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# TRAINING LAUNCHER
# Train each of the appliances and models analyzed in the project and described in config/settings.yaml

mkdir -p logs/train_logs
mkdir -p outputs/train_outputs && cd outputs/train_outputs

################ DISHWASHER 

# Experiment 1
mkdir dishwasher
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher --path outputs/train_outputs/dishwasher --train --epochs 5 --disable-random > logs/train_logs/dishwasher-result.log

# Experiment 2
mkdir dishwasher-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-norm --path outputs/train_outputs/dishwasher-norm --train --epochs 5 --disable-random > logs/train_logs/dishwasher-norm-result.log

# Experiment 4
mkdir dishwasher-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-onlyregression --path outputs/train_outputs/dishwasher-onlyregression --train --epochs 5 --disable-random > logs/train_logs/dishwasher-onlyregression-result.log

# Experiment 5
mkdir dishwasher-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-onlyregression-norm --path outputs/train_outputs/dishwasher-onlyregression-norm --train --epochs 5 --disable-random > logs/train_logs/dishwasher-onlyregression-norm-result.log

# Experiment 7
mkdir dishwasher-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-classattention --path outputs/train_outputs/dishwasher-classattention --train --epochs 5 --disable-random > logs/train_logs/dishwasher-classattention-result.log

################ FRIDGE 

# Experiment 1
mkdir fridge
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge --path outputs/train_outputs/fridge --train --epochs 5 --disable-random > logs/train_logs/fridge-result.log

# Experiment 2
mkdir fridge-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-norm --path outputs/train_outputs/fridge-norm --train --epochs 5 --disable-random > logs/train_logs/fridge-norm-result.log

# Experiment 4
mkdir fridge-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-onlyregression --path outputs/train_outputs/fridge-onlyregression --train --epochs 5 --disable-random > logs/train_logs/fridge-onlyregression-result.log

# Experiment 5
mkdir fridge-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-onlyregression-norm --path outputs/train_outputs/fridge-onlyregression-norm --train --epochs 5 --disable-random > logs/train_logs/fridge-onlyregression-norm-result.log

# Experiment 7
mkdir fridge-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-classattention --path outputs/train_outputs/fridge-classattention --train --epochs 5 --disable-random > logs/train_logs/fridge-classattention-result.log

################ MICROWAVE

# Experiment 1
mkdir microwave
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave --path outputs/train_outputs/microwave --train --epochs 5 --disable-random > logs/train_logs/microwave-result.log

# Experiment 2
mkdir microwave-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-norm --path outputs/train_outputs/microwave-norm --train --epochs 5 --disable-random > logs/train_logs/microwave-norm-result.log

# Experiment 4
mkdir microwave-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-onlyregression --path outputs/train_outputs/microwave-onlyregression --train --epochs 5 --disable-random > logs/train_logs/microwave-onlyregression-result.log

# Experiment 5
mkdir microwave-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-onlyregression-norm --path outputs/train_outputs/microwave-onlyregression-norm --train --epochs 5 --disable-random > logs/train_logs/microwave-onlyregression-norm-result.log

# Experiment 7
mkdir microwave-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-classattention --path outputs/train_outputs/microwave-classattention --train --epochs 5 --disable-random > logs/train_logs/microwave-classattention-result.log

