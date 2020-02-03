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



if __name__ == '__main__':
	
	I = imageio.imread('male_0000.png')
	print(I.shape)

	I_gradient = sobel(I)
	I_gradient02 = sobel_fromCV(I)

	fig = plt.figure()
	ax01 = fig.add_subplot(131)
	ax01.imshow(I)

	ax02 = fig.add_subplot(132)
	ax02.imshow(I_gradient)

	ax03 = fig.add_subplot(133)
	ax03.imshow(I_gradient02)

	plt.show()


