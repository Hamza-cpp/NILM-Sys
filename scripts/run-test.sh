#
# TESTING LAUNCHER
# Test each of the appliances and models analyzed in the project and described in config/settings.yaml
# See documentation describing each of the appliance analyzed 
# See documentation describing each of the model architectures evaluated

mkdir -p logs/test_logs
mkdir -p outputs/test_outputs && cd outputs/test_outputs

############### DISHWASHER

# Experiment 1
mkdir dishwasher
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher --path outputs/test_outputs/dishwasher  --epochs 1 --disable-random > logs/test_logs/dishwasher-result.log

# Experiment 2
mkdir dishwasher-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-norm --path outputs/test_outputs/dishwasher-norm  --epochs 1 --disable-random > logs/test_logs/dishwasher-norm-result.log

# Experiment 3
mkdir dishwasher-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-norm --path outputs/test_outputs/dishwasher-norm-trainnorm  --epochs 1 --disable-random > logs/test_logs/dishwasher-norm-trainnorm-result.log

# Experiment 4
mkdir dishwasher-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-onlyregression --path outputs/test_outputs/dishwasher-onlyregression  --epochs 1 --disable-random > logs/test_logs/dishwasher-onlyregression-result.log

# Experiment 5
mkdir dishwasher-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-onlyregression-norm --path outputs/test_outputs/dishwasher-onlyregression-norm  --epochs 1 --disable-random > logs/test_logs/dishwasher-onlyregression-norm-result.log

# Experiment 5
mkdir dishwasher-onlyregression-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-onlyregression-norm --path outputs/test_outputs/dishwasher-onlyregression-norm-trainnorm  --epochs 1 --disable-random > logs/test_logs/dishwasher-onlyregression-norm-trainnorm-result.log

# Experiment 7
mkdir dishwasher-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance dishwasher-classattention --path outputs/test_outputs/dishwasher-classattention  --epochs 1 --disable-random > logs/test_logs/dishwasher-classattention-result.log

################ FRIDGE

# Experiment 1
mkdir fridge
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge --path outputs/test_outputs/fridge  --epochs 1 --disable-random > logs/test_logs/fridge-result.log

# Experiment 2
mkdir fridge-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-norm --path outputs/test_outputs/fridge-norm  --epochs 1 --disable-random > logs/test_logs/fridge-norm-result.log

# Experiment 3
mkdir fridge-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-norm --path outputs/test_outputs/fridge-norm-trainnorm  --epochs 1 --disable-random > logs/test_logs/fridge-norm-trainnorm-result.log

# Experiment 4
mkdir fridge-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-onlyregression --path outputs/test_outputs/fridge-onlyregression  --epochs 1 --disable-random > logs/test_logs/fridge-onlyregression-result.log

# Experiment 5
mkdir fridge-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-onlyregression-norm --path outputs/test_outputs/fridge-onlyregression-norm  --epochs 1 --disable-random > logs/test_logs/fridge-onlyregression-norm-result.log

# Experiment 6
mkdir fridge-onlyregression-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-onlyregression-norm --path outputs/test_outputs/fridge-onlyregression-norm-trainnorm  --epochs 1 --disable-random > logs/test_logs/fridge-onlyregression-norm-trainnorm-result.log

# Experiment 7
mkdir fridge-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance fridge-classattention --path outputs/test_outputs/fridge-classattention  --epochs 1 --disable-random > logs/test_logs/fridge-classattention-result.log

################# MICROWAVE

# Experiment 1
mkdir test_outputs/microwave
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave --path outputs/test_outputs/microwave  --epochs 1 --disable-random > logs/test_logs/microwave-result.log

# Experiment 2
mkdir microwave-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-norm --path outputs/test_outputs/microwave-norm  --epochs 1 --disable-random > logs/test_logs/microwave-norm-result.log

# Experiment 3
mkdir microwave-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-norm --path outputs/test_outputs/microwave-norm-trainnorm  --epochs 1 --disable-random > logs/test_logs/microwave-norm-trainnorm-result.log

# Experiment 4
mkdir microwave-onlyregression
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-onlyregression --path outputs/test_outputs/microwave-onlyregression  --epochs 1 --disable-random > logs/test_logs/microwave-onlyregression-result.log

# Experiment 5
mkdir microwave-onlyregression-norm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-onlyregression-norm --path outputs/test_outputs/microwave-onlyregression-norm  --epochs 1 --disable-random > logs/test_logs/microwave-onlyregression-norm-result.log

# Experiment 6
mkdir microwave-onlyregression-norm-trainnorm
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-onlyregression-norm --path outputs/test_outputs/microwave-onlyregression-norm-trainnorm  --epochs 1 --disable-random > logs/test_logs/microwave-onlyregression-norm-trainnorm-result.log

# Experiment 7
mkdir microwave-classattention
CUDA_VISIBLE_DEVICES=0 python -u main.py --settings config/settings.yaml --appliance microwave-classattention --path outputs/test_outputs/microwave-classattention  --epochs 1 --disable-random > logs/test_logs/microwave-classattention-result.log
