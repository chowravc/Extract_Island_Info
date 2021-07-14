## Importing useful packages
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
import os
import shutil
import pims
import csv
import trackpy as tp

## Importing useful scripts
from utils import *



## Remove all touching islands in each image
def remove_touching_all(directory, tolerance=3):

	# Reading output files
	filesList = glob.glob(directory + 'labels/*.txt')

	# If at least one file is found
	if len(filesList) > 0:

		# Folders to put outputted non-touching islands
		folderNT = directory + 'labels_nt/'

		# Make folder for non-touching
		if len(glob.glob(folderNT)) == 1:
			shutil.rmtree(folderNT)
		os.mkdir(folderNT)

		# Center of the bubble on image
		bubble_center = [495, 618]

		# List of points at the edge of the bubble
		bubble_edges = [[495, 141], [21, 618], [947, 618], [110, 337], [826, 929], [291, 185]]

		# Predict the radius of the bubble
		radius = get_bubble_radius(bubble_center, bubble_edges)

		# For every file in filesList
		for i, labelFile in enumerate(filesList):

			print('\nRemoving touching from file: ' + labelFile)

			# Run remove touching
			remove_touching_ff(labelFile, radius, tolerance=tolerance)



## Create pandas dataframe for trackpy
def to_pandas_df(directory, absolute=False, cutoff=0):

	# Output dataframes
	outPD_path = directory + 'dataframe.csv'

	# Read all inputs
	filesList = glob.glob(directory + 'labels_nt/*.txt')

	# Output dictionary
	outPD = {
		'x': [],
		'y': [],
		'mass': [],
		'size': [],
		'ecc': [],
		'signal': [],
		'raw_mass': [],
		'ep': [],
		'frame': []
	}

	# As long as at least one file exists
	if len(filesList) > 0:

		# For every file
		for i, labelFile in enumerate(filesList):

			# Display current file
			print("Doing file: " + labelFile)

			if str(cutoff) in labelFile:

				break

			# Load label file as numpy array
			labelMat = np.loadtxt(labelFile)

			# For every island in array
			for j, island in enumerate(labelMat):

				# Format:
				# calc_radius, calc_area, calc_phi, theta, lmbda, xc, yc, w, h
				# 0 - calc_radius
				# 1 - calc_area
				# 2 - calc_phi
				# 3 - theta
				# 4 - lmbda
				# 5 - xc
				# 6 - yc
				# 7 - w
				# 8 - h

				# If you want it in absolute (3d) coordinates
				if absolute:

					# This one is in absolute (3d) coordinates
					outPD['x'].append(island[3]) # Theta
					outPD['y'].append(island[4]) # Lambda
					outPD['mass'].append(island[1])	# Area (absolute)
					outPD['size'].append(island[1]) # Area (absolute)
					outPD['ecc'].append(0.0) # Eccentricity (absolute, circles)
					outPD['signal'].append(10.0) # ?
					outPD['raw_mass'].append(island[1]) # Area (absolute)
					outPD['ep'].append(0.1) # ?
					outPD['frame'].append(i) # Frame number

				# If you want the apparent (2d image) coordinates
				else:

					# This one is in absolute (3d) coordinates
					outPD['x'].append(island[5]) # xc
					outPD['y'].append(island[6]) # yc
					outPD['mass'].append(island[1])	# Area (absolute)
					outPD['size'].append(island[1]) # Area (absolute)
					outPD['ecc'].append(0.0) # Eccentricity (absolute, circles)
					outPD['signal'].append(10.0) # ?
					outPD['raw_mass'].append(island[1]) # Area (absolute)
					outPD['ep'].append(0.1) # ?
					outPD['frame'].append(i) # Frame number



	# Convert the dictionary to a pandas dataframe
	outPD = pd.DataFrame.from_dict(outPD)

	# Save the dataframe as csv
	outPD.to_csv(outPD_path)



## Link the islands from a csv pandas dataframe
def link_islands(csv_file, dist, memory):

	# Read unlinked csv file
	unlinked = pd.read_csv(csv_file)

	# Name for output linked dataframe
	csv_out = csv_file.replace('dataframe', 'linked')

	# Link the islands in the dataframe
	linked = tp.link(unlinked, dist, memory=memo)

	# Save the dataframe as csv
	linked.to_csv(csv_out)



## Display trajectories
def disp_trajectories(csv_file):

	# Read linked csv file
	linked = pd.read_csv(csv_file)

	# Plot trajectories
	tp.plot_traj(linked);



## Plot velocity field
def plot_velo(inDir, csv_file, frame=1000, delta=5):

	# Read linked csv file
	linked = pd.read_csv(csv_file)

	# Path to frame
	framePath = inDir + 'frames/' + str(frame).zfill(4) + '.tif'

	# Image read
	im = Image.open(framePath)

	# Store matrix of values for current frame
	labelMatCurrent = []
	# Store matrix of values for frame + delta
	labelMatDelta = []

	# Store plot lists in this dictionary
	toPlot = {
		'x': [],
		'y': [],
		'dx': [],
		'dy': []
	}

	# Open linked csv file
	with open(csv_file, 'r') as data:

		# Go through it linewise
		for i, line in enumerate(csv.reader(data)):

			# If it is from the right frame
			if line[-2] == str(frame):

				# Add the line
				labelMatCurrent.append(np.asarray(line))

			# If it is from frame + delta
			if line[-2] == str(frame + delta):

				# Add the line
				labelMatDelta.append(np.asarray(line))

	# For each line in the current frame
	for l1 in labelMatCurrent:

		# Extract particle number
		currentPart = l1[-1]

		# Try to find the same particle in the frame+delta
		for l2 in labelMatDelta:

			# If it is found
			if l2[-1] == currentPart:

				# Store coordinates in frame
				toPlot['x'].append(int(float(l1[2])))
				toPlot['y'].append(int(float(l1[3])))

				# Store relative motion
				toPlot['dx'].append(-int(float(l1[2])) + int(float(l2[2])))
				toPlot['dy'].append(int(float(l1[3])) - int(float(l2[3])))

	# Plot the frame image
	plt.imshow(im)

	# Plot the velocity field
	plt.quiver(toPlot['x'], toPlot['y'], toPlot['dx'], toPlot['dy'], pivot='middle', linewidths=1, headwidth=4, headlength=6, color='red')
	
	# Turn off axis
	plt.axis('off')

	# Save the figure with no white space
	plt.savefig(csv_file.replace('csv', 'tif').replace('linked', 'velo'), bbox_inches = 'tight', pad_inches=0, dpi=400)

	# Show the figure
	plt.show()



## Plot all islands from linked on frame
def plot_frame(csv_file, inDir, frame=0):

	# Framepath
	framePath = inDir + 'frames/' + str(frame).zfill(4) + '.tif'

	# Open linked csv file
	with open(csv_file, 'r') as data:

		# Lists to store values
		islands, xs, ys = [], [], []

		# Go through it linewise
		for i, line in enumerate(csv.reader(data)):

			# Header looks like this
			# ['', 'Unnamed: 0', 'x', 'y', 'mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep', 'frame', 'particle']
		
			# Display current line
			if i%10000 == 0:

				print('Currently on line: ' + str(i))

			# The first line contains the headers (keys)
			if i != 0:

				# This is which frame it is
				cF = line[-2]

				# If it is from the correct frame
				if cF == str(frame):

					# Get the island
					islands.append(line[-1])

					# Get the coordinates
					xs.append(float(line[2]))
					ys.append(float(line[3]))

	print('Found ' + str(len(xs)) + ' nontouching islands.')

	# Read image
	im = Image.open(framePath)

	# Display image in the background
	plt.imshow(im)

	# For every island in the frame
	for i in range(len(islands)):

		# Display the label associated with particle
		plt.text(x=xs[i]+0.1, y=ys[i]+0.1, s=islands[i], fontdict=dict(color='red', size=8))

	# Plot dots on each particle
	plt.scatter(xs, ys, color='r', s=2)

	# Show plot
	plt.show()



## Plot diameters changing over latitude
def plot_diam(csv_file, iPlot, degree):

	# Center of the bubble on image
	bubble_center = [495, 618]

	# List of points at the edge of the bubble
	bubble_edges = [[495, 141], [21, 618], [947, 618], [110, 337], [826, 929], [291, 185]]

	# Predict the radius of the bubble
	radius = get_bubble_radius(bubble_center, bubble_edges)

	# Output plot name
	plot_path = csv_file.replace('csv', 'tif').replace('linked', 'diameters')

	# Dictionary to store what has to be displayed
	dictPlot = {
		'lats': [],
		'diams': []
	}

	# Open linked csv file
	with open(csv_file, 'r') as data:

		# Go through it linewise
		for i, line in enumerate(csv.reader(data)):

			if i%10000 == 0:

				print('Currently on line: ' + str(i))

			# The first line contains the headers (keys)
			if i != 0:

				# This is which island it is
				island = int(line[-1])

				# If the current island needs to be plotted
				if island in iPlot:

					coords = spherical(bubble_center, radius, (float(line[2]), float(line[3])))
				
					# Current latitude (this is the x-coord)
					cLat = (-1)*coords[1]*180/np.pi + 90

					# Diameter
					cD = 2*np.sqrt((float(line[4])/np.pi))

					# Add them to the dictionary
					dictPlot['lats'].append(cLat)
					dictPlot['diams'].append(cD)

	# Display how many positions are there in total
	print('Number of points discovered: ' + str(len(dictPlot['lats'])))

	# First point
	fP = [dictPlot['lats'][0], dictPlot['diams'][0]]

	# Last point
	lP = [dictPlot['lats'][-1], dictPlot['diams'][-1]]

	# Perform a fit
	fitCoeffs = np.polyfit(dictPlot['lats'], dictPlot['diams'], degree)

	# Function to run fit
	fit = np.poly1d(fitCoeffs)

	print('Degree of poly fit: ' + str(degree))

	# Create a numpy list of latitudes
	fitLats = np.linspace(fP[0], lP[0], num=50)

	# Find corresponding diameters according to fit
	fitDiams = fit(fitLats)

	# Clear plot
	plt.clf()

	# Plot the points
	plt.scatter(dictPlot['lats'], dictPlot['diams'], s=2, color='b')

	# Plot the fit
	plt.plot(fitLats, fitDiams, color='k')

	# Plot the rough start and end points
	plt.scatter(fP[0], fP[1], s=40, color='g')
	plt.scatter(lP[0], fP[1], s=40, color='r')

	plt.title('Island diameter vs Latitude (recorded). Island: ' + repr(iPlot))
	plt.xlabel('Lat (degrees, equator at 0)')
	plt.ylabel('Diameter (pixels)')

	# Show cumulative plot
	plt.savefig('temp_points.tif', dpi=300)
	plt.show()

	# Create new numpy array of latitudes
	fitLats = np.linspace(fP[0], lP[0], num=10)

	# Find corresponding diameters according to fir
	fitDiams = fit(fitLats)

	# Clear plots again
	plt.clf()

	# Plot the fit
	plt.scatter(fitLats, fitDiams, color='k', s=40)

	plt.title('Island diameter vs Latitude (fit points). Island: ' + repr(iPlot))
	plt.xlabel('Lat (degrees, equator at 0)')
	plt.ylabel('Diameter (pixels)')

	# Show cumulative plot
	plt.savefig('temp_fit.tif', dpi=300)
	plt.show()


## Main function of the file
def main(directory, display=False, showIsland=False):

	# Reading input label files
	filesList = glob.glob(directory + 'labels/*.txt')

	# If at least one file found
	if len(filesList) > 0:

		# Make folder for outputs
		if len(glob.glob('output/')) == 0:
			os.mkdir('output/')

		# Output folder
		outDir = directory.replace('input', 'output')

		# Make folder for specific run
		if len(glob.glob(outDir)) == 1:
			shutil.rmtree(outDir)
		os.mkdir(outDir)
		os.mkdir(outDir + 'labels/')
		os.mkdir(outDir + 'frames/')

		# Dimensions of the image
		dims = (1024, 1024)

		# Center of the bubble on image
		bubble_center = [495, 618]

		# List of points at the edge of the bubble
		bubble_edges = [[495, 141], [21, 618], [947, 618], [110, 337], [826, 929], [291, 185]]

		# Predict the radius of the bubble
		radius = get_bubble_radius(bubble_center, bubble_edges)

		# Doing every file individually
		for i, labelFile in enumerate(filesList):

			print("\nDoing " + labelFile + ', ' + str(i+1) + '/' + str(len(filesList)))

			# Calculate predictions for radii of each island
			island_sizes(labelFile, dims, bubble_center, radius, display=display, showIsland=showIsland)



## If file is run directly
if __name__ == '__main__':

	####
	# Main function
	# Input directory
	# inDir = 'input/309_tm_25C_top_need_35C/'
	# inDir = 'input/309_tm_26C_bot_need_40C/'
	# inDir = 'input/309_tm_26C_bot_need_35C/'
	# inDir = 'input/309_tm_26C_bot_need_45C/'
	# inDir = 'input/309_tm_28C_bot_need_50C/'

	# Run main function
	# main(inDir, False, False)

	####
	# Remove touching from output
	# touchDir = 'output/309_tm_25C_top_need_35C/'
	# touchDir = 'output/309_tm_26C_bot_need_40C/'
	# touchDir = 'output/309_tm_26C_bot_need_35C/'
	# touchDir = 'output/309_tm_26C_bot_need_45C/'
	# touchDir = 'output/309_tm_28C_bot_need_50C/'

	# Tolerance for touching
	tol = 3

	# Run remove touching function
	# remove_touching_all(touchDir, tol)

	####
	# Saving pandas dataframe for trackpy

	# Directory
	# outdir = 'output/309_tm_25C_top_need_35C/'
	# outdir = 'output/309_tm_26C_bot_need_40C/'
	# outdir = 'output/309_tm_26C_bot_need_35C/'
	# outdir = 'output/309_tm_26C_bot_need_45C/'
	# outdir = 'output/309_tm_28C_bot_need_50C/'

	absolute = False

	# Run function to make dataframe
	# to_pandas_df(outdir, absolute, cutoff=9999)

	####
	# Linking islands

	# Path to pandas dataframe
	# pathPD = 'output/309_tm_25C_top_need_35C/dataframe.csv'
	# pathPD = 'output/309_tm_26C_bot_need_40C/dataframe.csv'
	# pathPD = 'output/309_tm_26C_bot_need_35C/dataframe.csv'
	# pathPD = 'output/309_tm_26C_bot_need_45C/dataframe.csv'
	# pathPD = 'output/309_tm_28C_bot_need_50C/dataframe.csv'

	# Run functions to link islands
	dis = 5
	memo = 3

	# link_islands(pathPD, dist=dis, memory=memo)

	####
	# Displaying trajectories

	# Path to linked pandas dataframe
	# pathPD_linked = 'output/309_tm_25C_top_need_35C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_40C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_35C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_45C/linked.csv'
	# pathPD_linked = 'output/309_tm_28C_bot_need_50C/linked.csv'

	# Call function
	# disp_trajectories(pathPD_linked)

	####
	# Plotting instantaneous velocity

	# Path to input directory
	# inDir = 'input/309_tm_25C_top_need_35C/'
	# inDir = 'input/309_tm_26C_bot_need_40C/'
	# inDir = 'input/309_tm_26C_bot_need_35C/'
	# inDir = 'input/309_tm_26C_bot_need_45C/'
	# inDir = 'input/309_tm_28C_bot_need_50C/'

	# Path to linked pandas dataframe
	# pathPD_linked = 'output/309_tm_25C_top_need_35C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_40C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_35C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_45C/linked.csv'
	# pathPD_linked = 'output/309_tm_28C_bot_need_50C/linked.csv'

	frame = 100
	delta = 50

	# Call function
	# plot_velo(inDir, pathPD_linked, frame, delta)

	####
	# Displaying frame with labelled islands

	# Linked csv file
	# pathPD_linked = 'output/309_tm_25C_top_need_35C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_40C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_35C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_45C/linked.csv'
	# pathPD_linked = 'output/309_tm_28C_bot_need_50C/linked.csv'

	# Path to input directory
	# inDir = 'input/309_tm_25C_top_need_35C/'
	# inDir = 'input/309_tm_26C_bot_need_40C/'
	# inDir = 'input/309_tm_26C_bot_need_35C/'
	# inDir = 'input/309_tm_26C_bot_need_45C/'
	# inDir = 'input/309_tm_28C_bot_need_50C/'

	# Frame to display
	# f = int(input('Frame: '))

	# Call function
	# plot_frame(pathPD_linked, inDir, frame=f)

	####
	# Plotting change in diameter

	# Path to linked pandas dataframe
	# pathPD_linked = 'output/309_tm_25C_top_need_35C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_40C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_35C/linked.csv'
	# pathPD_linked = 'output/309_tm_26C_bot_need_45C/linked.csv'
	# pathPD_linked = 'output/309_tm_28C_bot_need_50C/linked.csv'

	# Island(s) to plot diameter vs lat
	islands = [410, 1218]

	# Degree of polynomial to fit
	degree = 3

	# Call function
	# plot_diam(pathPD_linked, islands, degree)


	#### List of islands to plot Diameter vs Latitude
	#
	# Note: Within the list of islands, everything inside ' ' is a single island which may have been misrecognized as two.
	#		Islands are labelled top to bottom, left to right.
	#
	# 1. island siza vs _latitude 309_tm_26C_bot_need_35C.png
	#	 Frame: 0443
	#	 Number of islands: 1
	#	 bottom
	#	 Islands in linked csv: '661, 2575, 2651, 2733'
	#
	# 2. island siza vs _latitude 309_tm_26C_bot_need_40C.png
	#	 Frame: 0341
	#	 Number of islands: 2
	#	 bottom
	#	 Islands in linked csv: '404, 1719', '543'
	#
	# 3. island siza vs _latitude309_tm_25C_need35C.png
	#	 Frame: 1360
	#	 Number of islands: 10
	#	 bottom
	#	 Islands in linked csv: '267, 434, 617, 1481', '305, 1315', '307, 1162', '292', '36', '25, 888', '471, 1043', '42', '44', '339, 1045, 1627'
	#
	# 4. island siza vs _latitude309_tm_26C_bot_need_45C.png
	#	 Frame: 0381
	#	 Number of islands: 2
	#	 bottom
	#	 Islands in linked csv: '365, 1085', '410, 1218'