echo $1 as base directory
python scripts/unit_tau_concepts.py --dataset broden "$1" alexnet
python scripts/unit_tau_concepts.py --dataset broden "$1" resnet18
python scripts/unit_tau_concepts.py --dataset broden "$1" densenet161
python scripts/unit_tau_concepts.py --dataset imagenet "$1" alexnet
python scripts/unit_tau_concepts.py --dataset imagenet "$1" resnet18
python scripts/unit_tau_concepts.py --dataset imagenet "$1" densenet161
python scripts/direction_tau_concepts.py --dataset broden "$1" alexnet
python scripts/direction_tau_concepts.py --dataset broden "$1" resnet18
python scripts/direction_tau_concepts.py --dataset broden "$1" densenet161
python scripts/direction_tau_concepts.py --dataset imagenet "$1" alexnet
python scripts/direction_tau_concepts.py --dataset imagenet "$1" resnet18
python scripts/direction_tau_concepts.py --dataset imagenet "$1" densenet161
