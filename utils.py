## Importing useful packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ps
from PIL import Image


## Calculate radius of the bubble from the position of the center, and a list of points at the edge of the bubble
def get_bubble_radius(bubble_center, bubble_edges):

	# Storing radius values
	radii = []

	# For every point at the edge
	for edge_point in bubble_edges:

		# Calculate the position relative to center of bubble
		diff = [edge_point[0] - bubble_center[0], edge_point[1] - bubble_center[1]]

		# Find absolute distance (radius)
		diff = np.sqrt(np.sum([d**2 for d in diff]))

		# Append the calculated radius to list
		radii.append(diff)

	# Find average radius
	radius = np.mean(radii)

	# Find deviation in radius values
	rad_dev_of_mean = np.std(radii)/np.sqrt(len(radii))

	# Display results
	print("Bubble radius: " + str(radius))
	print("Uncertainty: " + str(100*rad_dev_of_mean/radius)[:4] + "%")

	# Return results
	return radius



## Convert YOLO labelling of bounding box (decimal) to absolute in pixels
def label_to_bb(label, dims):

	# Center of bounding box
	xc, yc = int(label[1]*dims[0]), int(label[2]*dims[1])

	# Height and width of bounding box
	w, h = int(label[3]*dims[0]), int(label[4]*dims[1])

	# Return tuple of pixelwise format
	return (xc, yc, w, h)



## Predicting the radius of an island from the bounding box, its location and information on the bubble
def island_radius(globe_center, globe_radius, bb_center, bb_dims):

	# Location of center of bounding box with respect to the center of the bubble
	dx = globe_center[0] - bb_center[0]
	dy = globe_center[1] - bb_center[1]

	# Absolute distance of bounding box center from center of bubble
	dr = np.sqrt(dx**2 + dy**2)

	# Just so that there is no divide by zero error
	if dx == 0:
		dx = 1

	# Angle of the center of the bounding box with respect to center of bubble
	phi = np.arctan(dy/dx)

	# Top left and bottom right of bounding box
	x0, y0 = bb_center[0] - bb_dims[0]/2, bb_center[1] - bb_dims[1]/2
	x1, y1 = bb_center[0] + bb_dims[0]/2, bb_center[1] + bb_dims[1]/2

	# Numerator in the equation to calculate radius
	rad_num = ((bb_dims[1]/2)**2) - ((bb_dims[0]/2)**2)*((dy/dx)**2)

	# Denominator in the equation to calculate radius
	rad_den = (np.cos(phi)**2) - (np.sin(phi)**2)*((dy/dx)**2)

	# Exceptions
	if rad_den == 0 or rad_num*rad_den < 0:
		print('Invalid bounding box.')
		return 0, phi, dr
	else:
		# Calculating radius
		radius = np.sqrt(rad_num/rad_den)

	# Outputting predicted radius and phi
	return radius, phi, dr


## Label all island radii and sizes with a YOLOv5 label file
def island_sizes(label_file, dims, bubble_center, bubble_radius, display=False, showIsland=False):

	# Generate ID for output file
	output_file = 'output/' + label_file.split('/')[-2] + '/' + label_file.split('/')[-1]

	# Load labels from input file
	labelMat = np.loadtxt(label_file)

	# Image displaying island path
	imagePath = label_file.replace('labels', 'frames').replace('txt', 'tif')

	# Reading image
	image = Image.open(imagePath)

	# Clear plots
	plt.clf()

	# First subplot
	plt.subplot(1, 2, 1)

	# Displaying original image
	plt.imshow(image)
	plt.title('Original Image')

	# Second subplot
	plt.subplot(1, 2, 2)
	plt.title('Image with overlaid island')
	plt.imshow(image)
	ax = plt.gca()

	# For every bounding box in the file
	for i, det in enumerate(labelMat):

		# Get the absolute pixelwise 
		xc, yc, w, h = label_to_bb(det, dims)

		# Predict the radius of the island, get angle and distance from center
		calc_radius, calc_phi, calc_dist = island_radius(bubble_center, bubble_radius, (xc, yc), (w, h))

		# Calculate area from predicted area
		calc_area = np.pi*calc_radius**2

		# Display results
		if display:
			print("\nIsland " + str(i+1) + '/' + str(len(labelMat)))
			print('Predicted island center: (' + str(xc) + ', ' + str(yc) + ')')
			print("Predicted island radius: " + str(calc_radius)[:5] + " px")
			print("Predicted island area: " + str(calc_area)[:7] + " sq. px")

		island_center = (xc, yc)
		ell_width = 2*calc_radius*np.cos(np.arcsin(calc_dist/bubble_radius))
		ell_height = 2*calc_radius

		ax.add_patch(ps.Ellipse(island_center,
			width=ell_width,
			height=ell_height,
			angle=calc_phi*(180/np.pi), 
			edgecolor='red',
			facecolor='none',
			linewidth=1))

	if show_island:
		plt.show()


## Running this if the script is run directly
if __name__ == '__main__':

	# Testing file
	file = 'input/309_tm_25C_top_need_35C/labels/0020.txt'

	# Dimensions of the image
	dims = (1024, 1024)

	# Center of the bubble on image
	bubble_center = [495, 618]

	# List of points at the edge of the bubble
	bubble_edges = [[495, 141], [21, 618], [947, 618]]

	# Predict the radius of the bubble
	radius = get_bubble_radius(bubble_center, bubble_edges)

	# Calculate predictions for radii of each island
	island_sizes(file, dims, bubble_center, radius, display=True, showIsland=True)