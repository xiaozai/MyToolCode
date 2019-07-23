import numpy as np 
import imageio
import random

"""
	2019.07.23, Song Yan
	Write two small scripts used for visualize segmentation results and depth map
"""

def save_depth_img(depth_mat, output_path='depth_img.png', default_max_bg=1e10, bg=-1):
	"""
		To convert the depth_mat to grey scaled image

		depth_mat.     : W*H matrix, contains depth information 
		default_max_bg : the default value for background, usually 1e10 (very large)
		bg             : set the bg value to a value which is closed to the foreground value
		output_path.   : the path to save the image

		return the grayscale image

	"""
	depth_im = depth_mat
	depth_im[depth_mat == default_max_bg] = bg
	depth_im = (depth_im - depth_im.min()) / depth_im.max() * 255
	depth_im = depth_im.astype(np.uint8)
	imageio.imwrite(output_path, depth_im)

	return depth_im

def save_segment_img(segm_mat, output_path='segm_img.png'):
	"""
		To convert the segm_mat into the RGB image with colored parts

		segm_mat       : W*H matrix, in which N labels (here is 10)
		output_path.   : the path to save the image
		Assign different colors for each part

	"""
	img = np.zeros((segm_mat.shape[0], segm_mat.shape[1], 3), dtype=np.uint8)

	img[segm_mat==0, :] = [  0,   0,   0]
	img[segm_mat==1, :] = [255,   0,   0]
	img[segm_mat==2, :] = [255, 255,   0]
	img[segm_mat==3, :] = [255,   0, 255]
	img[segm_mat==4, :] = [255, 255, 255]
	img[segm_mat==5, :] = [  0, 255,   0]
	img[segm_mat==6, :] = [  0, 255, 255]
	img[segm_mat==7, :] = [  0, 255, 194]
	img[segm_mat==8, :] = [168, 255,   0]
	img[segm_mat==9, :] = [ 45, 255,  75]

	imageio.imwrite(output_path, img)

	return img

#----------------------------------------------------------------------

if __name__ == '__main__':

	# Assume that depth_mat contains the depth , and backgournd is 1e10
	depth_mat = np.ones((640, 480), dtype=float)*1e10
	depth_mat[300:456, 123:345] = [[random.random() for i in range(123, 345)] for j in range(300, 456)]
	depth_im = save_depth_img(depth_mat, output_path='depth_img.png')
	#
	# Assume that segm_mat contains the segmentation results, 
	# in which there are 10 labels, 0 for bachground, others for parts
	# assign colors for each part
	segm_mat = np.asarray([[random.randint(0, 5) for i in range(0,480)] for j in range(0, 640)])
	segm_im  = save_segment_img(segm_mat, output_path='segm_img.png')


