"""This file contains dataset stats information"""


IMAGENET_STATS = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

CIFAR_STATS = {
    'mean': [0.491, 0.482, 0.447],
    'std': [0.247, 0.243, 0.261]
}

DATASET_STATS = {
    'cifar10': CIFAR_STATS,
    'cifar20': CIFAR_STATS,
    'stl10': IMAGENET_STATS
}
