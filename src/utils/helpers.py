from collections import defaultdict


def outer_default():
    return defaultdict(inner_default)


def inner_default():
    return {'points': 0, 'goals': 0}