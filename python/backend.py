import importlib
import numpy
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os
import argparse
import time
import datetime
import subprocess
import random
import string
import itertools
import heapq



import_cupy=False
import_torch=True

if import_cupy:
    try:
        be_np = importlib.import_module('cupy')
        be_scp = importlib.import_module('cupyx.scipy')
        be_scp_sig = importlib.import_module('cupyx.scipy.signal')
    except ImportError:
        be_np = importlib.import_module('numpy')
        be_scp = importlib.import_module('scipy')
        be_scp_sig = importlib.import_module('scipy.signal')
else:
    be_np = importlib.import_module('numpy')
    be_scp = importlib.import_module('scipy')
    be_scp_sig = importlib.import_module('scipy.signal')

if import_torch:
    import torch
    from torch import nn, optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    import torchvision.transforms as transforms


fft = be_np.fft.fft
fftshift = be_np.fft.fftshift

randn = be_np.random.randn
rand = be_np.random.rand
randint = be_np.random.randint
uniform = be_np.random.uniform
normal = be_np.random.normal
choice = be_np.random.choice
exponential = be_np.random.exponential

firwin = be_scp_sig.firwin
lfilter = be_scp_sig.lfilter
freqz = be_scp_sig.freqz
welch = be_scp_sig.welch
upfirdn = be_scp_sig.upfirdn
constants = be_scp.constants
chi2 = be_scp.stats.chi2


