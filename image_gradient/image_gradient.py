import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt 

def sobel(img):
	'''
	sobel kernel
	   gx =[ +1 0 -1     gy =[ +1 +2 +1
	   	     +2 0 -2            0  0  0
		     +1 0 -1 ]         -1 -2 -1 ]
	
	GX = gx * A,   GY = gy * A,  * is convolution operation
	G = square(GX^2 + GY^2)

	'''

	# reverse , using dot product instead of convolution
	Gx = np.array([[-1, 0, 1],
		  		   [-2, 0, 2],
		  		   [-1, 0, 1]]) 
	Gy = np.array([[-1, -2, -1],
		  		   [ 0,  0,  0],
		  		   [ 1,  2,  1]])

	rows = img.shape[0]
	cols = img.shape[1]
	mag = np.zeros((rows, cols), dtype=np.float32)

	for i in range(rows-2):
		for j in range(cols-2):
			S1 = np.sum(Gx*img[i:i+3, j:j+3])
			S2 = np.sum(Gy*img[i:i+3, j:j+3])

			mag[i+1, j+1] = np.sqrt(S1**2 + S2**2)
	
	return mag

def sobel_fromCV(img):
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	mag = np.sqrt(sobelx**2 + sobely**2)

	return mag

# def getContour(img):
# 	s = np.linspace(0, 2*np.pi, 400)
# 	r = 100 + 100*np.sin(s)
# 	c = 220 + 100*np.cos(s)
# 	init = np.array([r, c]).T

# 	snake = active_contour(gaussian(img, 3), 
# 						   init, 
# 						   alpha=0.015, 
# 						   beta=10, 
# 						   gamma=0.001, 
# 						   coordinates='rc')
# 	return snake

def getContourPts(img):
	H, W = img.shape

	pts = []
	for jj in range(H):
		for ii in range(W):
			if img[jj, ii] > 0:
				pts.append([ii, jj])
	pts = np.array(pts)
	return pts

if __name__ == '__main__':
	
	I = imageio.imread('male_0000.png')

	# I_gradient = sobel(I)
	I_gradient02 = sobel_fromCV(I)
	pts = getContourPts(I_gradient02)
	print(pts.shape)

	# fig, ax = plt.subplots()
	# ax.imshow(I)
	# ax.plot(pts[:, 0], pts[:, 1], 'k.', lw=0.1)
	# ax.set_xlim([0, 200])
	# ax.set_ylim([480, 0])
	# plt.show()

	fig = plt.figure()
	ax01 = fig.add_subplot(131)
	ax01.imshow(I)

	ax01.set_ylim([480, 0])
	ax01.set_xlim([0, 200])
	ax01.plot(pts[:, 0], pts[:, 1], 'k.', lw=0.1)

	ax02 = fig.add_subplot(132)
	# ax02.imshow(I_gradient)
	ax02.plot(pts[:, 0], pts[:, 1], 'b.', lw=0.1)
	ax02.set_ylim([480, 0])
	ax02.set_xlim([0, 200])

	ax03 = fig.add_subplot(133)
	ax03.imshow(I_gradient02)
	ax03.set_ylim([480, 0])
	ax03.set_xlim([0, 200])

	plt.show()


