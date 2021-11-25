import os

MODALITY='M'  #RGB:M, Depth:K
GPU_IDS=0
FRAME=32

os.system('python ./AutoGesture/Single_modality/train_AutoGesture_3DCDC.py -m {} -t {} -g {} --config {} | tee {}'.format(
    'valid', MODALITY, GPU_IDS, './AutoGesture/Single_modality/config.yml',f'./AutoGesture/Single_modality/log/model-{MODALITY}-{FRAME}.log'
))