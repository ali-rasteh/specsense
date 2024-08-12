import importlib
from sklearn.linear_model import Ridge
import numpy
import matplotlib.pyplot as plt
import argparse
import time
import os


import_cupy=False
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


fft = be_np.fft.fft
fftshift = be_np.fft.fftshift

randn = be_np.random.randn
rand = be_np.random.rand
uniform = be_np.random.uniform
normal = be_np.random.normal

firwin = be_scp_sig.firwin
lfilter = be_scp_sig.lfilter
freqz = be_scp_sig.freqz
welch = be_scp_sig.welch
consts = be_scp.constants
