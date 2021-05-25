import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from utils import (calculate_ssim, psnr, rmse, normalize, normalize_all)
import cv2

def plot_flame(ori_img, rec_img, savepath):
    L = ori_img.shape[2]
    point = [[133,88],[133,103],[133,118],[133,133]]
    x = [i for i in range(400,640+1,5)]
    plt.figure(1)
    plt.imshow(ori_img[:,:,20],cmap=plt.cm.gray)
    plt.plot(point[0][0],point[0][1], marker='x', markersize=10, color='blue')
    plt.plot(point[1][0],point[1][1], marker='x', markersize=10, color='green')
    plt.plot(point[2][0],point[2][1], marker='x', markersize=10, color='red')
    plt.plot(point[3][0],point[3][1], marker='x', markersize=10, color='yellow')
    plt.savefig(savepath + './flame_spectral.png')

    plt.figure(2, figsize=(8,9))
    plt.subplot(411)
    plt.plot(x, normalize_all(ori_img[point[0][0],point[0][1],:]), label='origin', color='blue', linestyle='--')
    plt.scatter(x, normalize_all(rec_img[point[0][0],point[0][1],:]), label='GAP-TV', color='orange', marker='o')
    plt.ylabel('Reflectance(blue)', fontsize=15)
    plt.title('GAP-TV flame reconstruction result', fontsize=15)
    plt.xticks([])
    plt.legend(fontsize=15,loc='upper left')
    plt.subplot(412)
    plt.plot(x, normalize_all(ori_img[point[1][0],point[1][1],:]), label='origin', color='blue', linestyle='--')
    plt.scatter(x, normalize_all(rec_img[point[1][0],point[1][1],:]), label='GAP-TV', color='orange', marker='o')
    plt.ylabel('Reflectance(green)', fontsize=15)
    plt.xticks([])
    plt.legend(fontsize=15,loc='upper left')
    plt.subplot(413)
    plt.plot(x, normalize_all(ori_img[point[2][0],point[2][1],:]), label='origin', color='blue', linestyle='--')
    plt.scatter(x, normalize_all(rec_img[point[2][0],point[2][1],:]), label='GAP-TV', color='orange', marker='o')
    plt.ylabel('Reflectance(red)', fontsize=15)
    plt.xticks([])
    plt.legend(fontsize=15,loc='upper left')
    plt.subplot(414)
    plt.plot(x, normalize_all(ori_img[point[3][0],point[3][1],:]), label='origin', color='blue', linestyle='--')
    plt.scatter(x, normalize_all(rec_img[point[3][0],point[3][1],:]), label='GAP-TV', color='orange', marker='o')
    plt.xlabel('Spectral bands(nm)', fontsize=15)
    plt.ylabel('Reflectance(yellow)', fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.show()


def plot_img(ori_img, rec_img, savepath, img_name):
    ori_img = normalize_all(ori_img)
    rec_img = normalize_all(rec_img)
    L = ori_img.shape[2]
    point = [[90,80],[90,160],[170,80],[170,160]]
    x = [i for i in range(400,700+1,10)]
    plt.figure(1)
    plt.imshow(ori_img[:,:,20],cmap=plt.cm.gray)
    plt.plot(point[0][0],point[0][1], marker='x', markersize=15, color='blue')
    plt.plot(point[1][0],point[1][1], marker='x', markersize=15, color='green')
    plt.plot(point[2][0],point[2][1], marker='x', markersize=15, color='red')
    plt.plot(point[3][0],point[3][1], marker='x', markersize=15, color='yellow')
    plt.axis('off')
    plt.savefig(savepath + f'./{img_name}_spectral.png')

    plt.figure(2, figsize=(8,9))
    plt.subplot(411)
    plt.plot(x, normalize_all(ori_img[point[0][0],point[0][1],:]), label='origin', color='blue', linestyle='--')
    plt.scatter(x, normalize_all(rec_img[point[0][0],point[0][1],:]), label='GAP-TV', color='orange', marker='o')
    plt.ylabel('Reflectance(blue)', fontsize=15)
    plt.title('GAP-TV image reconstruction result', fontsize=15)
    plt.xticks([])
    plt.legend(fontsize=15,loc='upper left')
    plt.subplot(412)
    plt.plot(x, normalize_all(ori_img[point[1][0],point[1][1],:]), label='origin', color='blue', linestyle='--')
    plt.scatter(x, normalize_all(rec_img[point[1][0],point[1][1],:]), label='GAP-TV', color='orange', marker='o')
    plt.ylabel('Reflectance(green)', fontsize=15)
    plt.xticks([])
    plt.legend(fontsize=15,loc='upper left')
    plt.subplot(413)
    plt.plot(x, normalize_all(ori_img[point[2][0],point[2][1],:]), label='origin', color='blue', linestyle='--')
    plt.scatter(x, normalize_all(rec_img[point[2][0],point[2][1],:]), label='GAP-TV', color='orange', marker='o')
    plt.ylabel('Reflectance(red)', fontsize=15)
    plt.xticks([])
    plt.legend(fontsize=15,loc='upper left')
    plt.subplot(414)
    plt.plot(x, normalize_all(ori_img[point[3][0],point[3][1],:]), label='origin', color='blue', linestyle='--')
    plt.scatter(x, normalize_all(rec_img[point[3][0],point[3][1],:]), label='GAP-TV', color='orange', marker='o')
    plt.xlabel('Spectral bands(nm)', fontsize=15)
    plt.ylabel('Reflectance(yellow)', fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.show()


if __name__ == '__main__':

    TYPE = 'CAVE' # ['CAVE','FLAME]

    if TYPE == 'FLAME':
        result_dir = './Flame_results'
        result_mat = sio.loadmat(result_dir+f'/flame_img_result.mat')['img']
        img_mat = sio.loadmat(f'./flame_img.mat')['img']
        result_mat[result_mat<=0] = 0
        PSNR = psnr(img_mat, result_mat)
        SSIM = calculate_ssim(img_mat, result_mat)
        RMSE = rmse(img_mat, result_mat)

        # print metrics
        print(f'>>>>>> Flame Results >>>>>>')
        print(f'PSNR: {PSNR} dB')
        print(f'SSIM: {SSIM}')
        print(f'RMSE: {RMSE}')
        
        plot = 0 # plot spectral comparison
        if plot: plot_flame(img_mat, result_mat, result_dir)

        save = 1 # save to txt/png
        if save:
            with open(result_dir + '/flame_result.txt','w') as f:
                f.write(f'>>>>>> Flame Results >>>>>>\n')
                f.write(f'PSNR: {PSNR} dB\n')
                f.write(f'SSIM: {SSIM}\n')
                f.write(f'RMSE: {RMSE}\n')
            L = img_mat.shape[2]
            for i in range(L):
                cv2.imwrite(result_dir + f'/flame_origin_all/flame_origin_{i+1}.png', img_mat[:,:,i]*255)
                cv2.imwrite(result_dir + f'/flame_result_all/flame_result_{i+1}.png', result_mat[:,:,i]*255)

    elif TYPE == 'CAVE':
        result_dir = './CAVE_results'
        name_list = [
            'sponges_ms',
            'glass_tiles_ms',
            'superballs_ms',
            'thread_spools_ms',
            'flowers_ms',
        ]
        img_name = name_list[4] # choose
        result_mat = sio.loadmat(result_dir+f'/{img_name}_result.mat')['img']
        img_mat = sio.loadmat(result_dir + f'/{img_name}.mat')['img']
        PSNR = psnr(img_mat, result_mat)
        SSIM = calculate_ssim(img_mat, result_mat)
        RMSE = rmse(img_mat, result_mat)

        # print metrics
        print(f'>>>>>> {img_name} Results >>>>>>')
        print(f'PSNR: {PSNR} dB')
        print(f'SSIM: {SSIM}')
        print(f'RMSE: {RMSE}')

        plot = 1 # plot spectral comparison
        if plot: plot_img(img_mat, result_mat, result_dir, img_name)

        save = 0 # save to txt/png
        if save:
            with open(result_dir + f'/{img_name}_result.txt','w') as f:
                f.write(f'>>>>>> {img_name} Results >>>>>>\n')
                f.write(f'PSNR: {PSNR} dB\n')
                f.write(f'SSIM: {SSIM}\n')
                f.write(f'RMSE: {RMSE}\n')
            L = img_mat.shape[2]
            for i in range(L):
                cv2.imwrite(result_dir + f'/{img_name}_origin_all/{img_name}_origin_{i+1}.png', img_mat[:,:,i]*255)
                cv2.imwrite(result_dir + f'/{img_name}_result_all/{img_name}_result_{i+1}.png', result_mat[:,:,i]*255)
        
    else:
        raise ValueError("nope!")