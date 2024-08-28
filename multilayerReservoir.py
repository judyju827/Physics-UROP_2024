"""Aims to reproduce the results of 'Neuromorphic Overparameterisation: Generalisation
and Few-Shot Learning in Multilayer Physical Neural
Networks' https://arxiv.org/pdf/2211.06373"""

import os
os.environ['HOTSPICE_USE_GPU'] = 'False'

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error

try: from context import hotspice
except ModuleNotFoundError: import hotspice