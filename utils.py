## Importing useful packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ps
from PIL import Image



## Remove touching islands from label file
def remove_touching_ff(label_file, globe_radius, tolerance=3):

	# Output file
	output_file = label_file.replace('labels', 'labels_nt')

	# Reading label_file
	labelMat = np.loadtxt(label_file)

	# Storing flagged touching islands
	flagged = []

	# For every island
	for i, i1 in enumerate(labelMat):

		# Choose another island
		for j, i2 in enumerate(labelMat):

			# They cannot be the same
			if i != j:

				# Read i1 useful data
				r1 = i1[0] # Radius
				c1 = [i1[3], i1[4]] # Spherical coords

				# Read i2 useful data
				r2 = i2[0] # Radius
				c2 = [i2[3], i2[4]] # Spherical coords

				# Get distance between islands
				dist = distance_on_globe(globe_radius, c1, c2)

				# Check if dist is lesser than sum of radii plus tolerance
				if dist < r1 + r2 + tolerance:

					# If not already flagged
					if i not in flagged:
						flagged.append(i)

					# If not already flagged
					if j not in flagged:
						flagged.append(j)

	# Storing output mat
	out_array = []

	# Now copy over all elements of label matrix, unless island is flagged
	for i, island in enumerate(labelMat):

		# If not flagged
		if i not in flagged:

			# Append island
			out_array.append(island)

	# Save the output
	np.savetxt(output_file, np.asarray(out_array))


## Find the distance between two points on the surface of a globe
def distance_on_globe(globe_radius, c1, c2):

	# c1 and c2 are (theta, lambda)

	# Convert thetas to longitude (the same)
	lon1 = c1[0]
	lon2 = c2[0]

	# Convert lambdas to latitude
	lat1 = c1[1] - np.pi/2
	lat2 = c2[1] - np.pi/2

	# Angular difference
	dlon = lon2 - lon1
	dlat = lat2 - lat1

	# Haversin function
	a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2

	# Finding angular distance
	c = 2*np.arcsin(np.sqrt(a))

	# Great circle distance
	d = globe_radius*c

	# Return result
	return d



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



## Get spherical coordinates
def spherical(globe_center, globe_radius, island_center):

	# Location of center of island with respect to the center of the bubble
	dx = globe_center[0] - island_center[0]
	dy = globe_center[1] - island_center[1]

	# Angle around the left and right
	theta = np.arcsin(dx/globe_radius)

	# Angle from top to bottom
	lmbda = np.arccos(dy/globe_radius)

	# Return resultant coordinates
	return theta, lmbda



## Label all island radii and sizes with a YOLOv5 label file
def island_sizes(label_file, dims, bubble_center, bubble_radius, display=False, showIsland=False):

	# Generate ID for output file
	output_file = label_file.replace('input', 'output')

	# Generate ID for output image
	output_image = output_file.replace('labels', 'frames').replace('txt', 'png')

	# Load labels from input file
	labelMat = np.loadtxt(label_file)

	# Output matrix of islands
	out_array = []

	# Image displaying island path
	imagePath = label_file.replace('labels', 'frames').replace('txt', 'tif')

	# Reading image
	image = Image.open(imagePath)

	# Clear plots
	plt.clf()

	# Title the plot
	plt.title('Image with overlaid islands')

	# Show image
	plt.imshow(image)
	ax = plt.gca()

	# For every bounding box in the file
	for i, det in enumerate(labelMat):

		# Get the absolute pixelwise 
		xc, yc, w, h = label_to_bb(det, dims)

		# Center of ellipse on the image
		island_center = (xc, yc)

		# Predict the radius of the island, get angle and distance from center
		calc_radius, calc_phi, calc_dist = island_radius(bubble_center, bubble_radius, island_center, (w, h))

		# Spherical coordinates of center of island
		theta, lmbda = spherical(bubble_center, bubble_radius, island_center)

		# Calculate area from predicted area
		calc_area = np.pi*calc_radius**2

		# Append the island info to array
		if calc_radius != 0:
			out_array.append(np.array([calc_radius, calc_area, calc_phi, theta, lmbda, xc, yc, w, h]))

		# Display results
		if display:
			print("\nIsland " + str(i+1) + '/' + str(len(labelMat)))
			print('Predicted island center: (' + str(xc) + ', ' + str(yc) + ')')
			print("Predicted island center spherical (\u03B8, \u03BB): (" + str(theta*(180/np.pi))[:5] + "\u00B0, " + str(lmbda*(180/np.pi))[:5] + "\u00B0)")
			print("Predicted island radius: " + str(calc_radius)[:5] + " px")
			print("Predicted island area: " + str(calc_area)[:7] + " sq. px")

		# Width of the ellipse (twice semiminor axis)
		ell_width = 2*calc_radius*np.cos(np.arcsin(calc_dist/bubble_radius))

		# Height of the ellipse (twice semimajor axis)
		ell_height = 2*calc_radius

		# Add the ellipse
		ax.add_patch(ps.Ellipse(island_center,
			width=ell_width,
			height=ell_height,
			angle=calc_phi*(180/np.pi), 
			edgecolor='red',
			facecolor='none',
			linewidth=1))

	# Save the numpy array
	np.savetxt(output_file, np.asarray(out_array))

	# Save the figure
	plt.savefig(output_image, dpi=300)

	# If you want to display
	if showIsland:

		# Show the figure
		plt.show()

	# Clear figure
	plt.clf()


## Running this if the script is run directly
if __name__ == '__main__':

	# Testing file
	file = 'input/309_tm_25C_top_need_35C/labels/0020.txt'

	# Dimensions of the image
	dims = (1024, 1024)

	# Center of the bubble on image
	bubble_center = [495, 618]

	# List of points at the edge of the bubble
	bubble_edges = [[495, 141], [21, 618], [947, 618], [110, 337], [826, 929], [291, 185]]

	# Predict the radius of the bubble
	radius = get_bubble_radius(bubble_center, bubble_edges)

	# Calculate predictions for radii of each island
	island_sizes(file, dims, bubble_center, radius, display=True, showIsland=False)