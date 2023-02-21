# Data Points Attraction Manipulation(DPAM) against Membership Inference Attack
## Experimental environment
The primary environment is Python 3.7 and you need to install pytorch==1.6.0, sklearn==1.0.2, opacus (for differential privacy, comment out the import of this library if you do not use differential privacy)
## Dataset
In this experiment, we used six data sets of MNIST, CIFAR10, CIFAR100, TEXAS, LOCATIONS, and PURCHASE for the experiment. Among them, MNIST and CIFAR10 are 10 classification data sets, LOCATIONS contains 30 classifications, and TEXAS and CIFAR100 contain 100 classifications.
