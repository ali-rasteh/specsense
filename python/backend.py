import_general=True
import_networking=False
import_matplotlib=True
import_numpy=True
import_scipy=True
import_cupy=False
import_cupyx=False
import_sklearn=False
import_cv2=False
import_torch=False
import_sionna=True
import_pynq=False
import_sivers=False
import_adafruit=False

be_np = None
be_scp = None


if import_general:
    import importlib
    import os
    import shutil
    import nbformat
    import copy
    import json
    import platform
    import argparse
    import time
    import datetime
    import subprocess
    import random
    import string
    import paramiko
    from types import SimpleNamespace
    import itertools
    import heapq
    import atexit
    import pickle

if import_networking:
    from scp import SCPClient
    import requests
    import socket

if import_matplotlib:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Wedge, Circle, FancyArrow
    # matplotlib.use('TkAgg')
    matplotlib.use('WebAgg')
    # matplotlib.use('Agg')
    import skimage.measure as measure

if import_numpy:
    import numpy
    be_np = importlib.import_module('numpy')

if import_scipy:
    be_scp = importlib.import_module('scipy')
    be_scp_sig = importlib.import_module('scipy.signal')

if import_cupy:
    try:
        be_np = importlib.import_module('cupy')
    except ImportError:
        be_np = importlib.import_module('numpy')

if import_cupyx:
    try:
        be_scp = importlib.import_module('cupyx.scipy')
        be_scp_sig = importlib.import_module('cupyx.scipy.signal')
    except ImportError:
        be_scp = importlib.import_module('scipy')
        be_scp_sig = importlib.import_module('scipy.signal')

if import_numpy or import_cupy:
    fft = be_np.fft.fft
    ifft = be_np.fft.ifft
    fftshift = be_np.fft.fftshift
    ifftshift = be_np.fft.ifftshift

    randn = be_np.random.randn
    rand = be_np.random.rand
    randint = be_np.random.randint
    uniform = be_np.random.uniform
    normal = be_np.random.normal
    choice = be_np.random.choice
    exponential = be_np.random.exponential

if import_scipy or import_cupyx:
    constants = be_scp.constants
    # chi2 = be_scp.stats.chi2
    stats = be_scp.stats

    firwin = be_scp_sig.firwin
    lfilter = be_scp_sig.lfilter
    filtfilt = be_scp_sig.filtfilt
    freqz = be_scp_sig.freqz
    welch = be_scp_sig.welch
    upfirdn = be_scp_sig.upfirdn
    convolve = be_scp_sig.convolve
    resample = be_scp_sig.resample

if import_sklearn:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

if import_cv2:
    import cv2

if import_torch:
    import torch
    from torch import nn, optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, random_split
    import torchvision.transforms as transforms
    from fvcore.nn import FlopCountAnalysis


if import_sionna:
    if os.getenv("CUDA_VISIBLE_DEVICES") is None:
        gpu_num = 0 # Use "" to use the CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import sionna
    import mitsuba as mi

    # Configure the notebook to use only a single GPU and allocate only as much memory as needed
    # For more details, see https://www.tensorflow.org/guide/gpu
    import tensorflow as tf
    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_memory_growth(gpus[0], True)
    #     except RuntimeError as e:
    #         print(e)
    tf.get_logger().setLevel('ERROR')

    from sionna.rt import load_scene, Camera, Transmitter, Receiver, PlanarArray,\
                        PathSolver, RadioMapSolver
    from sionna.phy import Block
    from sionna.phy.mimo import StreamManagement, lmmse_equalizer
    from sionna.phy.ofdm import ResourceGrid, ResourceGridMapper, BaseChannelEstimator, LSChannelEstimator, LMMSEEqualizer, OFDMEqualizer, \
                            OFDMModulator, OFDMDemodulator, RZFPrecoder, RemoveNulledSubcarriers
    from sionna.phy.channel.tr38901 import Antenna, AntennaArray, UMi, UMa, RMa, CDL
    from sionna.phy.channel import gen_single_sector_topology as gen_topology
    from sionna.phy.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, \
                               time_lag_discrete_time_channel, ApplyOFDMChannel, ApplyTimeChannel, \
                               OFDMChannel, TimeChannel, CIRDataset, ChannelModel
    from sionna.phy.fec.ldpc import LDPC5GEncoder, LDPC5GDecoder
    from sionna.phy.mapping import Mapper, Demapper, BinarySource, QAMSource
    from sionna.phy.utils import ebnodb2no, sim_ber, compute_ber, flatten_dims, split_dim, flatten_last_dims,\
                             expand_to_rank, inv_cholesky
    

if import_pynq:
    from pynq import Overlay, allocate, MMIO, Clocks, interrupt, GPIO
    from pynq.lib import dma
    import xrfclk
    import xrfdc

if import_sivers:
    from pyftdi.ftdi import Ftdi

if import_adafruit:
    import board
    from adafruit_motorkit import MotorKit
    from adafruit_motor import stepper

