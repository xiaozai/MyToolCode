import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt 

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage.filters import maximum_filter
import random 

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
	'''
	using the Sobel function provided by OpenCV
	'''
	sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
	sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
	mag = np.sqrt(sobelx**2 + sobely**2)

	return mag, sobelx, sobely 

def getEdgePts(img):
	'''
	obtain the points on the edge, 
	background = 0, edge points > 0
	'''

	# max pooling, because the sobel filters generates a complex edges
	# like    . . .                                     0 0 0
	#         . . .  -> the edge , after max pooling -> 0 1 0
	#         . . .                                     0 0 0
	# remove some redundent points
	# img = img*(img == maximum_filter(img,footprint=np.ones((5,5))))
	img = img*(img == maximum_filter(img,size=(3,3)))

	H, W = img.shape

	pts = []
	for jj in range(H):
		for ii in range(W):
			if img[jj, ii] > 0:
				pts.append([ii, jj])
	pts = np.array(pts)
	return pts

def TSP(pts):
	'''
	 travelling salesman problem:
		 Given a list of cities and the distances between each pair of cities, 
		 what is the shortest possible route 
		 that visits each city and returns to the origin city?
	'''

	orderedPts = []

	xmax = 224-1
	ymax = 480-1

	# pts = list(pts)
	minPtY = min(pts[:, 1])
	idx = np.where(pts[:,1] == minPtY)
	startIdx = idx[0][0]

	startPts = [pts[startIdx, 0], pts[startIdx, 1]]

	orderedPts.append(startPts)

	pts = pts.tolist() # N*2 , pts[i] = [x, y]
	pts.pop(startIdx)
	
	# fig = plt.figure()
	# plt.plot(startPts[0], startPts[1], 'k.')

	num_loops = len(pts) - 1

	for _ in range(num_loops):

		min_dist = 1e4
		next_pts = []
		next_idx = 0
		for pIdx in range(len(pts)):
			pp = pts[pIdx]
			dist = np.sqrt((startPts[0] - pp[0])**2 + (startPts[1] - pp[1])**2)
			if dist < min_dist:
				min_dist = dist 
				next_pts = pp
				next_idx = pIdx 

		# print(next_pts)
		# print(min_dist)
		# plt.plot(next_pts[0], next_pts[1], 'b.')
		# plt.plot([startPts[0], next_pts[0]], [startPts[1], next_pts[1]], 'g-')
		# plt.pause(0.1)

		orderedPts.append(next_pts)
		pts.pop(next_idx)
		startPts = next_pts 

	if len(pts) == 1:
		orderedPts.append(pts[0])

	# plt.show()

	orderedPts = np.array(orderedPts)
	return orderedPts

if __name__ == '__main__':
	
	I = imageio.imread('male_0000.png')

	# I_gradient = sobel(I)
	I_gradient02, sobelx, sobely = sobel_fromCV(I)

	# get the edge points from gradient
	pts = getEdgePts(I_gradient02)
	print(len(pts))
	
	# Order the pts
	op = TSP(pts)

	numPts = 300
	step = np.int(np.floor(len(op)  / numPts))
	idx = np.arange(0, len(op), step)

	op = op[idx, :]
	
	if len(op) > numPts:
		idx = np.arange(0, len(op))
		idx = np.sort(random.sample(idx, numPts))
		op = op[idx, :]

	print(len(op)) # 314

	fig = plt.figure()

	for ii in range(len(op)-1):
		curPt = op[ii]
		nexPt = op[ii+1]

		plt.plot(curPt[0], curPt[1], 'k.')
		plt.plot([curPt[0], nexPt[0]], [curPt[1], nexPt[1]], 'b-')
		plt.pause(0.1)

	plt.plot(nexPt[0], nexPt[1], 'k.')
	plt.show()

	
	# Show
	# fig = plt.figure()
	# ax01 = fig.add_subplot(141)
	# ax01.imshow(I_gradient02)

	# ax01.set_ylim([480, 0])
	# ax01.set_xlim([0, 200])
	# ax01.plot(pts[:, 0], pts[:, 1], 'k.', lw=0.1)

	# ax02 = fig.add_subplot(142)
	# ax02.imshow(sobelx)
	# ax02.set_ylim([480, 0])
	# ax02.set_xlim([0, 200])

	# ax03 = fig.add_subplot(143)
	# ax03.imshow(sobely)
	# ax03.set_ylim([480, 0])
	# ax03.set_xlim([0, 200])

	# ax04 = fig.add_subplot(144)
	# ax04.plot(op[:, 0], op[:,1], 'g-')
	# ax04.plot(op[:, 0], op[:,1], 'r.')
	# ax04.set_ylim([480, 0])
	# ax04.set_xlim([0, 200])

	# plt.show()
