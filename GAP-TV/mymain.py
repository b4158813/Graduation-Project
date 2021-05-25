import os
import time
import math
import numpy as np
from statistics import mean
import scipy.io as sio
import matplotlib.pyplot as plt
from my_dvp_linear_inv_cassi import gap_denoise
from utils import (A, At, psnr,shift,shift_back,CAVE2mat,get_mask3d)

np.random.seed(5)
# TYPE = 'CAVE'
TYPE = 'FLAME'

if TYPE == 'CAVE':
    name_list = [
        'sponges_ms',
        'glass_tiles_ms',
        'superballs_ms',
        'thread_spools_ms',
        'flowers_ms'
    ]
    png_name = name_list[4]
    datasetdir = f'../CAVE_dataset/{png_name}/{png_name}/' # dataset dir
    resultsdir = './CAVE_results' # results dir
    dataname = f'{png_name}' # name of the dataset
    CAVE2mat(datasetdir + dataname + '_', dataname) # create a .mat file which contains CAVE png
    R, C, L, step = 256, 256, 31, 1
else:
    dataname = 'flame_img'
    resultsdir = './Flame_results'
    R, C, L, step = 224, 224, 49, 1


# get mask_shift
mask_3d = get_mask3d(R,C,L,step)

# get truth_shift
truth = sio.loadmat(f'.\\{dataname}.mat')['img']
truth_shift = np.zeros((R, C + step * (L - 1), L))
for i in range(L):
    truth_shift[:,i*step:i*step+C,i]=truth[:,:,i]

# get measurement
meas = np.sum(np.multiply(mask_3d, truth_shift), 2) #与01编码孔径相乘，并沿光谱维度累加
plt.imshow(meas,cmap=plt.cm.gray)
plt.axis('off')
plt.savefig(f'.\\{resultsdir}\\{dataname}_meas.png')
Phi = mask_3d
# exit(0)

method = 'GAP'

if method == 'GAP':
    _lambda = 1 # regularization factor
    accelerate = True # enable accelerated version of GAP
    # total variation (TV)
    denoiser = 'tv'
    iter_max = 20 # maximum number of iterations
    tv_weight = 6 # TV denoising weight (larger for smoother but slower)
    tv_iter_max = 5 # TV denoising maximum number of iterations each
    begin_time = time.time()
    vgaptv,psnr_gaptv = gap_denoise(meas,Phi,A,At,_lambda, 
                        accelerate, denoiser, iter_max, 
                        tv_weight=tv_weight, 
                        tv_iter_max=tv_iter_max,
                        X_orig=truth,sigma=[150,140,130,130,130,130,130,130])
    end_time = time.time()
    vrecon = shift_back(vgaptv,step=1)
    tgaptv = end_time - begin_time
    print(f'GAP-{denoiser.upper()} PSNR {mean(psnr_gaptv):2.2f} dB, running time {tgaptv:.1f} seconds.')
else:
    print('please input correct method.')


sio.savemat(f'.\\{resultsdir}\\{dataname}_result.mat',{'img':vrecon})

fig = plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(vrecon[:,:,(i+1)*3], cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
plt.savefig(f'.\\{resultsdir}\\{dataname}_result.png')
