python train_shadow.py --dataset CIFAR100 --epochs 100
python train_target.py --dataset CIFAR100 --epochs 100 --defense None --train_size 40000 --valid_size 10000 --attack_size 10000
python train_target.py --dataset CIFAR100 --epochs 100 --defense OUR --parameter 0.001 --train_size 40000 --valid_size 10000 --attack_size 10000
python train_target.py --dataset CIFAR100 --epochs 100 --defense OUR --parameter 0.002 --train_size 40000 --valid_size 10000 --attack_size 10000
python train_target.py --dataset CIFAR100 --epochs 100 --defense OUR --parameter 0.003 --train_size 40000 --valid_size 10000 --attack_size 10000
python train_target.py --dataset CIFAR100 --epochs 100 --defense GAN --parameter 1.0 --train_size 40000 --valid_size 10000 --attack_size 10000
python train_target.py --dataset CIFAR100 --epochs 100 --defense GAN --parameter 2.0 --train_size 40000 --valid_size 10000 --attack_size 10000
python attack.apy --dataset CIFAR100 --defense None
python attack.apy --dataset CIFAR100 --defense OUR --parameter 0.001
python attack.apy --dataset CIFAR100 --defense OUR --parameter 0.002
python attack.apy --dataset CIFAR100 --defense OUR --parameter 0.003
python attack.apy --dataset CIFAR100 --defense GAN --parameter 1.0
python attack.apy --dataset CIFAR100 --defense GAN --parameter 2.0
