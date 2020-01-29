import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt 
import cv2 
import math

def loadObj(fname, isMeter=True):
	vert, vt, tris = [], [], []
	with open(fname, 'r') as fp:
		content = fp.readlines()
	content = [x.strip() for x in content]

	for l in content:
		CC = l.split(' ')
		if len(CC) > 0:
			if CC[0] == 'v': vert.append([np.float32(x) for x in CC[1:4]])
			elif CC[0] == 'vt': vt.append([np.float32(x) for x in CC[1:3]])
			elif CC[0] == 'f': # f v/vt v/vt v/vt
				t1 = np.int32(CC[1].split('/')[0]) - 1
				t2 = np.int32(CC[2].split('/')[0]) - 1 
				t3 = np.int32(CC[3].split('/')[0]) - 1
				tris.append([t1, t2, t3])
	vert = np.asarray(vert)
	vert = vert - np.mean(vert, 0)
	if isMeter:
		vert = vert * 1000 # from meter to mm

	return vert, np.asarray(vt), np.asarray(tris)

class MeshToIMGProjector():
	def __init__(self, image_size=(224, 224), focal_len=35, camera_dist=350):
		''' focal length distance from the image plane to the Center Of Projection '''
		self.image_H = image_size[0]
		self.image_W = image_size[1]
		self.focal_len = focal_len
		self.camera_dist = camera_dist

	def getHomogeneousCoordinates(self, pts):
		''' add one column to the original points
			[x, y, z] => [x, y, z, 1]
		'''
		num_pts = pts.shape[0]            # (N, 3)
		homoCoord = np.ones((num_pts, 4)) # (N, 4)
		homoCoord[:, :-1] = pts             
		return homoCoord

	def projection(self, pts, camera='ortho'):
		if pts.shape[1] == 3:
			pts = self.getHomogeneousCoordinates(pts)
		pts = np.transpose(pts) # (4, N)

		projMatrix = np.zeros((3,4), dtype=np.float32)
		projMatrix[0, 0] = 1.0
		projMatrix[1, 1] = 1.0
			
		if camera == 'weakPerspective':
			z_avg = np.mean(pts, 1)[2]
			projMatrix[2, 3] = -z_avg/self.focal_len # z_avg / f
		elif camera == 'perspective':
			projMatrix[2, 2] = -1.0/self.focal_len   # z / f
		else:
			projMatrix[2, 3] = -self.camera_dist / self.focal_len # orthogonal

		projPts = np.matmul(projMatrix, pts)
		projPts = np.transpose(projPts)
		projPts[:, 0] = projPts[:, 0] / projPts[:, 2] # [x, y, z/f] => (dx/z, fx/z)
		projPts[:, 1] = projPts[:, 1] / projPts[:, 2] #  

		return projPts[:, :2]

	def checkVisiblePtsInImg(self, pts):
		''' check Pts are in the Image area or Not'''
		num_pts = pts.shape[0]
		flags = np.full((num_pts, ), True, dtype=bool)
		flags = np.logical_and(flags, abs(pts[:, 0]) <= self.image_W/2.0)
		flags = np.logical_and(flags, abs(pts[:, 1]) <= self.image_H/2.0)

		return flags

	def projectToPixelsCoord(self, pts, tris, camera='ortho'):
		''' from world Cooordinates (x, y, z) to Pixel Coordinates (h, w) 
			Pixels' coordinates refer to is located in the upper-left corner of the image.
			Pixel scale, 1 pixel = scale_factor *1 mm , default=1
		'''
		num_pts = pts.shape[0]

		# projection:  orthogonal, weak-perspective, perspective
		projPts = self.projection(pts, camera=camera)
		insidePixels, insideBarrys, _, _, _ =  self.rasterization(projPts, tris)
		projPts = np.concatenate((projPts, insidePixels), axis=0)
		# From Screen Space to Raster Space, Pixel's coordinates are intergers
		# 1) Re-map P's coordinates in the range [0, 1], Normalized Device Coordinate
		P_normalized = np.zeros((num_pts, 2), dtype=np.float32)
		P_normalized[:, 0] = (projPts[:, 0] + self.image_W / 2.0) / self.image_W # Width
		P_normalized[:, 1] = (projPts[:, 1] + self.image_H / 2.0) / self.image_H # Height
		# 2) define the pixel coordinate in raster space
		P_raster = np.zeros((num_pts, 2), dtype=np.int32)
		P_raster[:, 0] = np.int32(np.floor(P_normalized[:, 0] * self.image_W)) # width
		P_raster[:, 1] = np.int32(np.floor(P_normalized[:, 1] * self.image_H)) # height

		return P_raster

	def projectPixelCoordsToImg(self, pixel_coords, pixel_values, depth_values):
		num_pts = pixel_coords.shape[0]
		image = np.zeros((self.image_H, self.image_W), dtype=np.float32)
		depth = np.ones((self.image_H, self.image_W), dtype=np.float32) * 1e4

		for pt_idx in range(num_pts):
			# Check visible parts, if projPt lies outside of the image, ignore it
			x, y = pixel_coords[pt_idx, 0], pixel_coords[pt_idx, 1]
			if x>=0 and x < self.image_W and y >=0 and y < self.image_H:
				d_value = depth_values[pt_idx]
				# Check the depth info, 
				if d_value < depth[y, x]:
					depth[y, x] = d_value
					image[y, x] = pixel_values[pt_idx] 
		return image

	def checkVisibleByDepth(self, pixel_coords, depth_values):
		num_pts = pixel_coords.shape[0]
		depth_map = np.ones((self.image_H, self.image_W), dtype=np.float32) * 1e4
		visible_idx = np.ones((self.image_H, self.image_W), dtype=np.int32) * -1

		for idx in range(num_pts):
			x, y = pixel_coords[idx, 0], pixel_coords[idx, 1]
			d_value = depth_values[idx]
			if d_value < depth_map[y, x]:
				depth_map[y, x] = d_value 
				visible_idx[y, x] = idx 

		return visible_idx

	def sortPtsInClockWise(self, pt0, pt1, pt2, vIdx):
		pts = np.array([pt0, pt1, pt2])
		# calculate the angles between verts and the center
		center = np.mean(pts, 0)
		vectors = pts - center
		# angle between the vectors and the x-axis, [-pi , pi]
		angle00 = math.atan2(vectors[0][1], vectors[0][1])
		angle01 = math.atan2(vectors[1][1], vectors[1][1])
		angle02 = math.atan2(vectors[2][1], vectors[2][1])
		angles = np.array([angle00, angle01, angle02])
		# sort pts by angles, from +pi to -pi, clock-wise
		sort_idx = np.flip(np.argsort(angles)) 
		pts = pts[sort_idx]
		return pts[0], pts[1], pts[2], vIdx[sort_idx]

	def getPixelsInTriangle(self, pt0, pt1, pt2):
		'''
		 solved by Edge function, 
		 	E01(P) = (P.x - pt0.x) * (pt1.y - pt0.y) - (P.y - pt0.y) * (pt1.x - pt0.x)
		 	E12(P) = (P.x - pt1.x) * (pt2.y - pt1.y) - (P.y - pt1.y) * (pt2.x - pt1.x)
		 	E20(P) = (P.x - pt2.x) * (pt0.y - pt2.y) - (P.y - pt2.y) * (pt0.x - pt2.x)
			if all E(P)s are positive (clock-wise). then P lies inside the triangles
		'''
		pixels = np.empty((0, 2), dtype=np.int32)
		barrys = np.empty((0, 3), dtype=np.float32)

		# find the Bound Box
		xmin = np.rint(min(pt0[0], pt1[0], pt2[0]))
		xmax = np.rint(max(pt0[0], pt1[0], pt2[0]))
		ymin = np.rint(min(pt0[1], pt1[1], pt2[1]))
		ymax = np.rint(max(pt0[1], pt1[1], pt2[1]))
		# loop to check if pixel lies inside of the triangle
		for px in range(xmin, xmax):
			for py in range(ymin, ymax):
				# Edge Function
				E01 = (px - pt0[0])*(pt1[1] - pt0[1]) - (py - pt0[1])*(pt1[0] - pt0[0])
				E12 = (px - pt1[0])*(pt2[1] - pt1[1]) - (py - pt1[1])*(pt2[0] - pt1[0])
				E20 = (px - pt2[0])*(pt0[1] - pt2[1]) - (py - pt2[1])*(pt0[0] - pt2[0])
				# if all edge functions return positive or equal to 0
				if E01 >=0 and E12 >= 0 and E20 >=0:
					P = np.array([[px, py]])
					pixels = np.concatenate((pixels, P), axis=0)
					# get barraycentric coords
					bc = self.getBarrycentricCoords(P, pt0, pt1, pt2)
					barrys = np.concatenate((barrys, bc), axis=0)

		return pixels, barrys

	def getBarrycentricCoords(self, p, pt0, pt1, pt2):
		''' 
		 Barrycentric Coordinates
			P = lambda0 * V0 + lambda1 * V1 + lambda2 * V2
			lambda0, lambda1, lambda2 belong to [0, 1]
			and lambda0 + lambda1 + lambda2 = 1
			if lambda = 0, then P lies on the edge
			
			tri_area = area_BCP + area_ACP + area_ABP
		'''
		tri_area = np.linalg.norm(np.cross((pt1 - pt0), (pt2 - pt0))) 
		if tri_area == 0: # it is not a triable but a point
			v , u, w = 1, 0, 0
		else:
			# triangle BCP corresponds to pt0
			tri_BCP = np.linalg.norm(np.cross((p - pt2), (pt1 - pt2))) 
			v = tri_BCP / tri_area
			# triangle ACP corresponds to pt1
			tri_ACP = np.linalg.norm(np.cross((pt2 - pt0), (p - pt0))) 
			w = tri_ACP / tri_area 
			# triangle ABP corresponds to pt2
			u = 1 - v - w
		return np.array([[v, w, u]])

	def rasterization(self, pixel_coords, tris, rgb_values=None, depth_values=None, uv_values=None):
		'''
		Rasterization is the process of determining which pixels are inside a triangle
		'''
		insidePixels = np.empty(shape=(0, 2), dtype=np.int32)
		insideBarrys = np.empty(shape=(0, 3), dtype=np.float32)

		RGB   = np.empty(shape=(0, 3), dtype=np.float32)
		UV    = np.empty(shape=(0, 2), dtype=np.float32)
		DEPTH = np.empty(shape=(0, 1), dtype=np.float32)

		for ii in range(tris.shape[0]):
			vIdx = tris[ii, :]   
			V0 = pixel_coords[vIdx[0], :] 
			V1 = pixel_coords[vIdx[1], :]
			V2 = pixel_coords[vIdx[2], :]
			# sort points in clock-wise
			V0, V1, V2, vIdx = self.sortPtsInClockWise(V0, V1, V2, vIdx)
			# get pixels inside the triangles and their barrycentric coordinates
			pixels, bcCoords = self.getPixelsInTriangle(V0, V1, V2)
			insidePixels = np.concatenate((insidePixels, pixels), axis=0)
			insideBarrys = np.concatenate((insideBarrys, bcCoords), axis=0)

			# calculate the rgb values, depth values
			if rgb_values is not None:
				pixel_rgb = rgb_values[vIdx, :] # (3,3)
				for bc in bcCoords:
					pt_rgb = np.matmul(bc, pixel_rgb)
					RGB = np.concatenate((RGB, pt_rgb), axis=0)

			if depth_values is not None:
				pixel_depth = depth_values[vIdx, :]
				for bc in bcCoords:
					pt_depth = np.matmul(bc, pixel_depth)
					DEPTH = np.concatenate(DEPTH, pt_depth, axis=0)

			if uv_values is not None:
				pixel_uv = vu_values[vIdx, :]
				for bc in bcCoords:
					pt_uv = np.matmul(bc, pixel_uv)
					UV = np.concatenate((UV, pt_uv), axis=0)

		return insidePixels, insideBarrys, RGB, DEPTH, UV

if __name__ == '__main__':

	object_dist = 350 # mm, the distance from camera to the object

	pts, vt, tris = loadObj('csr4001a_with_VT.obj', isMeter=True)
	pts[:, 0] = -pts[:, 0] # make the mesh face to the origin
	pts[:, 2] = -pts[:, 2] # x = -x , z = -z, rotate 180 degree around Y-axis
	pts[:, 2] = pts[:, 2] + object_dist # Assume the camera at the origin

	projector = MeshToIMGProjector(image_size=(224, 224), focal_len=35, camera_dist=object_dist)
	pixel_coords = projector.projectToPixelsCoord(pts, tris, camera='ortho')

	# insidePixels, insideBarrys, _, _, _ =  projector.rasterization(pixel_coords, tris)

	# pixel_coords = np.concatenate((pixel_coords, insidePixels), axis=0)

	depth_values = np.ones((pixel_coords.shape[0],), dtype=np.float32)*255
	pixel_values = np.ones((pixel_coords.shape[0],), dtype=np.float32)*255

	mask  = projector.projectPixelCoordsToImg(pixel_coords, pixel_values, depth_values)
	# u_map = projector.projectPixelCoordsToImg(pixel_coords, vt[:, 0], depth_values) 
	# v_map = projector.projectPixelCoordsToImg(pixel_coords, vt[:, 1], depth_values)
	# d_map = projector.projectPixelCoordsToImg(pixel_coords, depth_values, depth_values)
	
	# visible_idx = projector.checkVisibleByDepth(pixel_coords, depth_values)

	cv2.imwrite('mask.png', mask)
	# cv2.imwrite('u_map.png', np.uint8(u_map*255))
	# cv2.imwrite('v_map.png', np.uint8(v_map*255))
	# cv2.imwrite('d_map.png', d_map)
	# cv2.imwrite('visible_idx.png', visible_idx)

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.imshow(visible_idx)
	# ax.title.set_text('visible idx map')
	# ax.set_xlim([0, 224])
	# ax.set_ylim([224, 0])
	# plt.show()

	# plt.figure()
	# ax = plt.axes(projection='3d')
	# ax.plot_trisurf(pts[:, 0], pts[:,1], pts[:,2], triangles=tris)
	# ax.scatter(0, 0, 0, 'rx', marker=3) # camera
	# ax.plot([0, 0], [0, 0], [0, 350], 'r-')
	# ax.set_xlabel('x')
	# ax.set_ylabel('y')
	# ax.set_zlabel('z')
	# plt.show()

	# fig = plt.figure()
	# ax1 = fig.add_subplot(221)
	# ax1.imshow(mask)
	# ax1.title.set_text('mask')
	# ax1.set_xlim([0, 224])
	# ax1.set_ylim([224, 0])

	# ax4 = fig.add_subplot(222)
	# ax4.imshow(d_map)
	# ax4.title.set_text('depth_map (Z-value)')
	# ax4.set_xlim([0, 224])
	# ax4.set_ylim([224, 0])

	# ax2 = fig.add_subplot(223)
	# ax2.imshow(u_map)
	# ax2.title.set_text('U')
	# ax2.set_xlim([0, 224])
	# ax2.set_ylim([224, 0])

	# ax3 = fig.add_subplot(224)
	# ax3.imshow(v_map)
	# ax3.title.set_text('V')
	# ax3.set_xlim([0, 224])
	# ax3.set_ylim([224, 0])

	# plt.show()

	# add texture color to the image
	# texture = cv2.imread('texture.jpg')
	# textureH, textureW = texture.shape[0], texture.shape[1]
	# num_pts = pixel_coords.shape[0]
	# R_values = np.zeros((num_pts,), dtype=np.float32)
	# G_values = np.zeros((num_pts,), dtype=np.float32)
	# B_values = np.zeros((num_pts,), dtype=np.float32)

	# for idx in range(num_pts):
	# 	u = np.int32(np.floor(vt[idx, 0] * textureW))
	# 	v = np.int32(np.floor(vt[idx, 1] * textureH))
	# 	R_values[idx] = texture[u,v,0]
	# 	G_values[idx] = texture[u,v,1]
	# 	B_values[idx] = texture[u,v,2]

	# R_map = projector.projectPixelCoordsToImg(pixel_coords, R_values, depth_values)
	# G_map = projector.projectPixelCoordsToImg(pixel_coords, G_values, depth_values)
	# B_map = projector.projectPixelCoordsToImg(pixel_coords, B_values, depth_values)
	
	# R_map = np.uint8(R_map)
	# G_map = np.uint8(G_map)
	# B_map = np.uint8(B_map)
	# rgb = np.dstack((R_map, G_map, B_map))
	# # cv2.imwrite('rgb.jpg', rgb)

	# fig = plt.figure()
	# ax = fig.add_subplot(111)
	# ax.imshow(rgb)
	# ax.title.set_text('with texture')
	# ax.set_xlim([0, 224])
	# ax.set_ylim([224, 0])
	# plt.show()