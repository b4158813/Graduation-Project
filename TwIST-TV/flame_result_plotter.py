import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def normalize(x):
    # return x / np.max(x)
    return (x-np.min(x))/(np.max(x)-np.min(x))

def cal_psnr(x, y):
    r, c = x.shape
    x = normalize(x)
    y = normalize(y)
    return -10*np.log10(np.sum((x-y)**2)/r/c)

def plot_spectra(imgpos, posv, posh, color, xla=True, xtic=True,tit=True):
    hx = [i for i in range(1,49+1)]
    plt.subplot(imgpos)
    plt.plot(hx,normalize(truth[posv-1,posh-1]),label=f'Ground truth ({color})',linestyle='--',linewidth=3,alpha=0.8)
    plt.scatter(hx,normalize(res[posv-1,posh-1]),color='orange',label='Reconstruction',linewidth=3)
    plt.legend(fontsize=15)
    plt.grid(linestyle='--',alpha=0.5)
    if tit: plt.title('CASSI with TwIST Reconstruction results(flame)',fontsize=15)
    if xla: plt.xlabel('Spectral band',fontsize=15)
    if not xtic: plt.xticks([])
    plt.ylabel('Reflectance',fontsize=15)
    plt.xticks(size=15)
    plt.yticks(size=15)


if __name__ == '__main__':
        
    N = 1
    path = 'C:\\Users\\wlx\\Desktop\\本科毕设CASSI模拟部分\\my_CASSI'

    res_path = path + f'\\rec_flame_{N}.mat' # recontstruction flame
    res = sio.loadmat(res_path)['x_twist']
    ori_path = path + f'\\orig_flame_{N}.mat' # origin flame
    truth = sio.loadmat(ori_path)['origin_img']



    pos = [[131,76],[131,96],[131,116],[131,136],[90,110]]


    hx = [i for i in range(1,49+1)]

    # N = 1 # num of showing pics
    # plt.figure(1, figsize=(6,18))
    # for i in range(N):
    #     plt.subplot(N,2,2*i+1)
    #     plt.imshow(res[:,:,13+i],cmap='gray')
    #     plt.plot(posv1,posh1,marker='x',markersize=20,color='pink')
    #     plt.title(f"picnum: {13+i+1}")
    #     plt.axis('off')
    #     plt.subplot(N,2,2*i+2)
    #     plt.imshow(truth[:,:,13+i],cmap='gray')
    #     plt.plot(posv1,posh1,marker='x',markersize=20,color='pink')
    #     plt.title(f"picnum: {13+i+1}")
    #     plt.axis('off')

    n = 44
    plt.figure(1, figsize=(6,3))
    plt.subplot(121)
    plt.imshow(res[:,:,n-1],cmap='gray')
    plt.plot(pos[0][0],pos[0][1],marker='x',markersize=20,color='pink')
    plt.plot(pos[1][0],pos[1][1],marker='x',markersize=20,color='yellow')
    plt.plot(pos[2][0],pos[2][1],marker='x',markersize=20,color='red')
    plt.plot(pos[3][0],pos[3][1],marker='x',markersize=20,color='green')
    plt.plot(pos[4][0],pos[4][1],marker='x',markersize=20,color='blue')
    # plt.text(10,25,f"PSNR={cal_psnr(res[:,:,n-1],truth[:,:,n-1]):.3f}",fontsize=15,color='w')
    plt.title(f"Reconstruction flame\npicnum: {n}")
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(truth[:,:,n-1],cmap='gray')
    plt.plot(pos[0][0],pos[0][1],marker='x',markersize=20,color='pink')
    plt.plot(pos[1][0],pos[1][1],marker='x',markersize=20,color='yellow')
    plt.plot(pos[2][0],pos[2][1],marker='x',markersize=20,color='red')
    plt.plot(pos[3][0],pos[3][1],marker='x',markersize=20,color='green')
    plt.plot(pos[4][0],pos[4][1],marker='x',markersize=20,color='blue')
    plt.title(f"Origin flame\npicnum: {n}")
    plt.axis('off')

    # plt.savefig(path + '\\reconstruction_flame1_img.png', dpi=300)

    plt.figure(2, figsize=(8,12))
    plot_spectra(511,pos[0][0],pos[0][1],'pink',xla=0,xtic=1,tit=1)
    plot_spectra(512,pos[1][0],pos[1][1],'yellow',xla=0,xtic=1,tit=0)
    plot_spectra(513,pos[2][0],pos[2][1],'red',xla=0,xtic=1,tit=0)
    plot_spectra(514,pos[3][0],pos[3][1],'green',xla=0,xtic=1,tit=0)
    plot_spectra(515,pos[4][0],pos[4][1],'blue',xla=1,xtic=1,tit=0)

    # plt.savefig(path + '\\reconstruction_flame1_plot.png', dpi=600)


    plt.show()

