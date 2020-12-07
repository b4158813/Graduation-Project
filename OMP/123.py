import random
import numpy as np
import time
import sys
import cv2
import matplotlib.pyplot as plt
import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg

# OMP algorithm
def OMP(y,A,K):
	col = A.shape[1]
	residual = y
	ind = []
	for i in range(K):
		prod = np.fabs(np.dot(A.T,residual))
		pos = np.argmax(prod)
		ind.append(pos)
		a = np.dot(np.linalg.pinv(A[:,ind]),y)
		residual = y-np.dot(A[:,ind],a)
	
	Res = np.zeros((col,))
	Res[ind] = a
	# print(Res.shape)
	return Res

img = cv2.imread("lena.jpg")
img = img[128:128+256, 128:128+256, 0]

# cv2.imshow("img",img)
# cv2.waitKey(0)

# radom gaussian matrix ----> undersample
sample_rate = 0.7
N = 256
Phi = np.random.randn(int(sample_rate*N),N)
Phi /= np.linalg.norm(Phi)

# DCT sparse matrix ----> sparse base
Psi = np.zeros((N,N))
n = np.array(range(N))
for k in range(N):
	Psi[k,:] = (2/N)**0.5*np.cos((2*n+1)*k*np.pi/2/N)
Psi[0,:] /= 2**0.5
Psi = Psi.T

# measurement ----> undersample image
measure_mat = np.dot(Phi,img)

# K coefficient for N colomns
sparse_coe = np.zeros((N,N))

# θ = Φ ψ
Theta = np.dot(Phi,Psi)


time_consume = []
# OMP for every colomn
for k in range(10,110,10):
	st = time.perf_counter()
	for i in range(N):
		sparse_coe[:,i] = OMP(measure_mat[:,i],Theta,k)
	en = time.perf_counter()
	img_rec = np.dot(Psi,sparse_coe)
	img_rec /= img_rec.max()
	img_rec *= 255
	img_rec = img_rec.astype(np.uint8)
	# print(img_rec.shape)
	# print(img.shape)
	# cv2.imshow("img",img)
	# cv2.imshow("rate=%.1f,k=%d"%(sample_rate,k),img_rec)
	sys.stdout.writelines(f"rate={sample_rate:.1f},k={k} runtime:{en-st}s\n")
	time_consume.append(en-st)
cv2.waitKey(0)

plt.figure(figsize=(12,8))
# plt.plot([i for i in range(256)],img[:,0])
# plt.plot([i for i in range(256)],img_rec[:,0])
# plt.legend(['origin','reconstruction'],fontsize=15)
# plt.title('Reconstruction result for column 0 with sample_rate=%.1f'%sample_rate, fontsize=15)
plt.plot([i for i in range(10,110,10)],time_consume)
plt.xlabel('K coefficient',fontsize=15)
plt.ylabel('consuming time(s)',fontsize=15)
plt.title('Consuming time for different k value',fontsize=20)
plt.grid()
plt.legend(['sample_rate=%.1f'%sample_rate],fontsize=15)
plt.show()
