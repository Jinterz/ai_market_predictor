# audioop.py
# Dummy module for audio operations.
# All functions here are stubbed and will raise NotImplementedError if called.

def _raise(*args, **kwargs):
    raise NotImplementedError("audioop is not available in this Python build.")

def avg(fragment, width):
    return _raise()

def max(fragment, width):
    return _raise()

def minmax(fragment, width):
    return _raise()

def rms(fragment, width):
    return _raise()

def tomono(fragment, width, factor):
    return _raise()

def tolin(fragment, width, factor):
    return _raise()

def lin2lin(fragment, width, newwidth):
    return _raise()

def add(fragment1, fragment2, width):
    return _raise()

def bias(fragment, width, bias):
    return _raise()

def reverse(fragment, width):
    return _raise()

def cross(fragment1, fragment2, width):
    return _raise()