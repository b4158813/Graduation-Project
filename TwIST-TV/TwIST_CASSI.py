import sys
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import cv2
from utils import (CAVE2mat,normalize)
from time import perf_counter

def shift(inputs, step):
    row,col,L = inputs.shape
    output = np.zeros((row,col+step*(L-1),L))
    for i in range(L):
        output[:,i*step:i*step+col,i] = inputs[:,:,i]
    return output

def shift_back(inputs, step):
    row,col,L = inputs.shape
    for i in range(L):
        inputs[:,:,i] = np.roll(inputs[:,:,i],(-1)*step*i,axis=1)
    output = inputs[:,0:col-step*(L-1),:]
    return output

def soft(x, T):
    return np.sign(x)*np.maximum(0,np.abs(x)-T)

def l1(x):
    return np.sum(np.abs(x))

def wraparound(x, m):
    '''
    padding x to be suitable for circular 2d convolution
    x: origin img
    m: kernel
    '''
    mx, nx = x.shape
    mm, nm = m.shape
    if mm > mx or nm > nx:
        x = x.T
        mx, nx = x.shape
        if mm > mx or nm > nx:
            raise Exception('kernel size is larger than img size!')

    mo = (1+mm)>>1
    no = (1+nm)>>1
    ml = mo - 1
    nl = no - 1
    mr = mm - mo
    nr = nm - no
    me = mx - ml + 1
    ne = nx - nl + 1
    mt = mx + ml
    nt = nx + nl
    my = mx + mm - 1
    ny = nx + nm - 1

    y = np.zeros((my, ny))
    y[mo-1:mt, no-1:nt] = x # cntral region == x
    
    if ml >= 0:
        y[0:ml, no-1:nt] = x[me-1:mx, :]
        if nl > 0:
            y[0:ml, 0:nl] = x[me-1:mx, ne-1:nx]
        if nr > 0:
            y[0:ml, nt:ny] = x[me-1:mx, 0:nr]
    if mr >= 0 :
        y[mt:my, no-1:nt] = x[0:mr, :]
        if nl >= 0:
            y[mt:my, 0:nl] = x[0:mr, ne-1:nx]
        if nr >= 0:
            y[mt:my, nt:ny] = x[0:mr, 0:nr]
    if nl >= 0:
        y[mo-1:mt, 0:nl] = x[:, ne-1:nx]
    if nr >= 0:
        y[mo-1:mt, nt:ny] = x[:, 0:nr]
    
    return y


# def conv2c(x, h):
#     '''
#     calculate circular 2d convolution of x with kernel h
#     '''
#     x = wraparound(x,h)
#     r1,c1 = x.shape
#     r2,c2 = h.shape
#     r,c = r1-r2+1,c1-c2+1
#     y = np.zeros((r,c))
#     h = np.flipud(np.fliplr(h)) # 旋转180° == 水平翻转 + 垂直翻转
#     # 求卷积
#     for i in range(r):
#         for j in range(c):
#             y[i,j] = np.sum(np.multiply(x[i:i+r2,j:j+c2],h[:,:]))
#     return y

def Rfun(x, Phi):
    return np.sum(x*Phi, axis=2)

def RTfun(y, Phi):
    return np.multiply(np.repeat(y[:,:,np.newaxis],Phi.shape[2],axis=2), Phi)

def diffh(x):
    # return conv2c(x,np.matrix([0,1,-1]))
    T = np.matrix([0,1,-1])
    return np.diff(wraparound(x,T),axis=1)[:,:-1]

def diffv(x):
    # return conv2c(x,np.matrix([0,1,-1]).T)
    T = np.matrix([0,1,-1]).T
    return np.diff(wraparound(x,T),axis=0)[:-1,:]

def TVnormspectralimaging(x):
    L = x.shape[2]
    y = np.zeros(L)
    for i in range(L):
        tpdh = diffh(x[:,:,i])
        tpdv = diffv(x[:,:,i])
        y[i] = np.sum(np.sqrt(np.multiply(tpdh,tpdh)+np.multiply(tpdv,tpdv)))

    return np.sum(y)

def modulo(pn):
    R = np.sqrt(np.sum(np.multiply(pn,pn),axis=1)).reshape((-1,1))
    return np.tile(R,(1,2))

def projk(g, Lambda, opQ, opQt, nither):
    '''
    Chambolle projection's algorithm from
    "An Algorithm for Total Variation Minimization and Applications", 2004
    '''
    tau = 0.25
    uy, ux = g.shape
    pn = np.zeros((uy*ux, 2))
    for i in range(nither):
        S = opQ(-opQt(pn) - g/Lambda)
        # print(S.shape,modulo(S).shape)
        pn = (pn+tau*S)/(1+tau*modulo(S))
    u = -Lambda * opQt(pn)
    return u

def mycalltoTVnew(mycube, th, piter=4):
    L = mycube.shape[2]
    img_estimated = np.zeros_like(mycube)
    for i in range(L):
        x = mycube[:,:,i]
        uy,ux = x.shape
        # dh = lambda x: conv2c(x, np.matrix([1,-1,0]))
        # dv = lambda x: conv2c(x, np.matrix([1,-1,0]).T)
        # dht = lambda x: conv2c(x, np.matrix([0,-1,1]))
        # dvt = lambda x: conv2c(x, np.matrix([0,-1,1]).T)
        dh = lambda x: np.diff(wraparound(x,np.matrix([0,1,-1])),axis=1)[:,1:]
        dv = lambda x: np.diff(wraparound(x,np.matrix([0,1,-1]).T),axis=0)[1:,:]
        dht = lambda x: -np.diff(wraparound(x,np.matrix([0,1,-1])),axis=1)[:,:-1]
        dvt = lambda x: -np.diff(wraparound(x,np.matrix([0,1,-1]).T),axis=0)[:-1,:]
        vect = lambda x: x.ravel()
        opQ = lambda x: np.c_[vect(dh(x)),vect(dv(x))]
        opQt = lambda x: dht(x[:,0].reshape((uy,ux))) + dvt(x[:,1].reshape((uy,ux)))
        img_estimated[:,:,i] = x - projk(x,th/2,opQ,opQt,piter)

    return img_estimated

def get_TwIST_ab(xi_1=1e-4, xi_n=1):
    rho0 = (1-xi_1/xi_n)/(1+xi_1/xi_n)
    alpha = 2/(1+np.sqrt(1-rho0*rho0))
    beta = 2*alpha/(xi_1+xi_n)
    return alpha, beta

def TwIST(y, A, AT, tau=0.03, maxiter=50, tolA=1e-8):
    '''
    tau: regulariazation paramater

    maxiter: max iteration times

    tolA: stopping threshold 
    '''
    
    alpha, beta = get_TwIST_ab()
    
    Aty = AT(y)
    # psi = lambda x,T: soft(x,T)
    # phi = lambda x: l1(x)
    psi = lambda x,th: mycalltoTVnew(x,th)
    phi = lambda x: TVnormspectralimaging(x)
    
    x = Aty # set iniitial x

    max_tau = np.max(np.abs(Aty))

    nz_x = np.where(x!=0,1,0)
    num_nz_x = np.sum(nz_x)

    resid = y - A(x)
    prev_f = 0.5*np.sum(np.multiply(resid,resid)) + tau*phi(x)

    Iter = 1
    sys.stdout.writelines(f"Initial objective = {prev_f:10.6e}, nonzeros = {num_nz_x:7d}\n")

    IST_iters = 0
    TwIST_iters = 0
    xm1 = xm2 = x
    max_svd = 1

    while True:
        grad = AT(resid)
        while True:
            # sys.stdout.writelines("ye\n")
            x = psi(xm2+grad/max_svd, tau/max_svd)
            if IST_iters >= 2 or TwIST_iters != 0: # do TwIST
                xm1 = np.multiply(xm1,x!=0)
                xm2 = np.multiply(xm2,x!=0)
                xm1 = (1-alpha)*xm1 + (alpha-beta)*xm2 + beta*x
                resid = y - A(xm1)
                f = 0.5*np.sum(np.multiply(resid,resid)) + tau*phi(xm1)
                if f > prev_f:
                    TwIST_iters = 0
                else:
                    TwIST_iters += 1
                    IST_iters = 0
                    x = xm1
                    if TwIST_iters % 10000 == 0:
                        max_svd *= 0.9
                    break
            else: # do IST
                resid = y - A(x)
                f = 0.5*np.sum(np.multiply(resid,resid)) + tau*phi(x)
                if f > prev_f:
                    max_svd *= 2
                    sys.stdout.writelines(f"Incrementing S = {max_svd:2.2e}\n")
                    IST_iters = 0
                    TwIST_iters = 0
                else:
                    TwIST_iters += 1
                    break
        
        xm1 = xm2
        xm2 = x

        nz_x_prev = nz_x
        nz_x = np.where(x!=0,1,0)
        num_nz_x = np.sum(nz_x)
        # num_changes_active = np.sum(np.where(nz_x_prev!=nz_x,1,0))

        criterion = np.abs(f-prev_f)/prev_f

        if Iter > maxiter or criterion < tolA:
            break

        Iter += 1
        prev_f = f

        sys.stdout.writelines(f"Iteration={Iter:4d}, objective={f:9.5e}, nz={num_nz_x:7d}, criterion={criterion/tolA:7.3e}\n")

    return x


if __name__ == '__main__':

    np.random.seed(5)

    # TYPE = 'FLAME'
    TYPE = 'CAVE'
    if TYPE == 'CAVE':
        name_list = [
            'sponges_ms',
            'glass_tiles_ms',
            'superballs_ms',
            'thread_spools_ms',
            'flowers_ms',
        ]
        png_name = name_list[0]
        datasetdir = f'../CAVE_dataset/{png_name}/{png_name}/'
        resultsdir = './CAVE_results'
        dataname = f'{png_name}'
        
        # get .mat file
        CAVE2mat(datasetdir + dataname + '_', dataname)

        R,C,L,step = 256,256,31,1
    else:
        resultsdir = f'./Flame_results'
        dataname = 'flame_img'
        
        R,C,L,step = 224,224,49,1


    # get img_shift
    origin_img = sio.loadmat(f'.\\{dataname}.mat')['img']
    origin_img_shift = shift(origin_img,step)
    
    # get CA_shift
    coded_aperture = np.zeros((R,C))
    for i in range(R):
        for j in range(C):
            coded_aperture[i,j] = np.random.choice([0,1])
    coded_aperture = np.repeat(coded_aperture[:,:,np.newaxis],L,axis=2)
    coded_aperture_shift = shift(coded_aperture,step)
    
    # get measurement
    meas = np.sum(np.multiply(origin_img_shift,coded_aperture_shift),axis=2)
    meas /= np.max(meas)
    plt.imshow(meas,cmap=plt.cm.gray)
    plt.axis('off')
    plt.savefig(f'.\\{resultsdir}\\{dataname}_meas.png')
    
    # operate TwIST
    AT = lambda x: RTfun(x,coded_aperture_shift)
    A = lambda x: Rfun(x,coded_aperture_shift)
    st = perf_counter()
    rec_x = TwIST(meas, A, AT)
    en = perf_counter()
    print(f'TIME: {en-st}s')
    rec_x = shift_back(rec_x, step)
    for i in range(L):
        rec_x[:,:,i] = rec_x[:,:,i]/np.max(rec_x[:,:,i])

    sio.savemat(f'.\\{resultsdir}\\{dataname}_result.mat',{'img': rec_x})

    fig = plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i+1)
        plt.imshow(rec_x[:,:,(i+1)*3], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
    plt.savefig(f'.\\{resultsdir}\\{dataname}_result.png')

