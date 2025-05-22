#!/bin/bash
nohup python -u ss_detection_test.py > ./logs/log_ss.log 2>&1 &
# nohup python -u fir_separate_test.py > ./logs/log_fil.log 2>&1 &
