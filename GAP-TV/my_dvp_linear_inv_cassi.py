import time
import math
import numpy as np
# from skimage.restoration import denoise_tv_chambolle
from utils import (A, At, psnr, shift, shift_back,calculate_ssim,TV_denoiser)
import scipy.io as sio


def gap_denoise(y, Phi, A, At, _lambda=1, accelerate=True, 
                denoiser='tv', iter_max=50, noise_estimate=True, sigma=None, 
                tv_weight=0.1, tv_iter_max=5, multichannel=True, x0=None, 
                X_orig=None, model=None, show_iqa=True):

    # [0] initialization
    if x0 is None:
        print(At)
        x0 = At(y, Phi) # default start point (initialized value)
    if not isinstance(sigma, list):
        sigma = [sigma]
    if not isinstance(iter_max, list):
        iter_max = [iter_max] * len(sigma)
    y1 = np.zeros_like(y)
    Phi_sum = np.sum(Phi,2)
    Phi_sum[Phi_sum==0]=1
    # [1] start iteration for reconstruction
    x = x0 # initialization
    psnr_all = []
    ssim_all=[]
    k = 0
    for idx, nsig in enumerate(sigma): # iterate all noise levels
        for it in range(iter_max[idx]):
            #print('max1_{0}_{1}:'.format(idx,it),np.max(x))
            yb = A(x,Phi)
            if accelerate: # accelerated version of GAP
                y1 = y1 + (y-yb)
                x = x + _lambda*(At((y1-yb)/Phi_sum,Phi)) # GAP_acc
            else:
                x = x + _lambda*(At((y-yb)/Phi_sum,Phi)) # GAP
            x = shift_back(x,step=1)
            # switch denoiser
            if denoiser.lower() == 'tv': # total variation (TV) denoising
                # x = denoise_tv_chambolle(x, nsig / 255, n_iter_max=tv_iter_max, multichannel=True)
                x= TV_denoiser(x, tv_weight, n_iter_max=tv_iter_max)
            else:
                raise ValueError('Unsupported denoiser {}!'.format(denoiser))
            # [optional] calculate image quality assessment, i.e., PSNR for 
            # every five iterations
            if show_iqa and X_orig is not None:
                ssim_all.append(calculate_ssim(X_orig, x))
                psnr_all.append(psnr(X_orig, x))
                if (k+1)%1 == 0:
                    if not noise_estimate and nsig is not None:
                        if nsig < 1:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}/255, ' 
                              'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, nsig*255, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
                        else:
                            print('  GAP-{0} iteration {1: 3d}, sigma {2: 3g}, ' 
                                'PSNR {3:2.2f} dB.'.format(denoiser.upper(), 
                                k+1, nsig, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
                    else:
                        print('  GAP-{0} iteration {1: 3d}, ' 
                              'PSNR {2:2.2f} dB.'.format(denoiser.upper(), 
                               k+1, psnr_all[k]),
                              'SSIM:{}'.format(ssim_all[k]))
            x = shift(x,step=1)
            if k==123:
                break
            k = k+1

    return x, psnr_all