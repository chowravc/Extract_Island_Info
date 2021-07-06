## Importing useful packages
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np

## Importing useful scripts
from utils import *

## Main function of the file
def main(directory):

	# Reading input label files
	filesList = glob.glob(directory + 'labels/*.txt')

	# If at least one file found
	if len(filesList) > 0:

		# Dimensions of the image
		dims = (1024, 1024)

		# Center of the bubble on image
		bubble_center = [495, 618]

		# List of points at the edge of the bubble
		bubble_edges = [[495, 141], [21, 618], [947, 618]]

		# Predict the radius of the bubble
		radius = get_bubble_radius(bubble_center, bubble_edges)

		# Doing every file individually
		for i, labelFile in enumerate(filesList):

			print("\nDoing " + labelFile + ', ' + str(i) + '/' + str(len(filesList)))

			# Calculate predictions for radii of each island
			island_sizes(labelFile, dims, bubble_center, radius, display=True)

			# # Plotting
			# x0, y0 = xc - w//2, yc - h//2
			# x1, y1 = xc + w//2, yc + h//2

			# bounds = (x0, y0, x1, y1)

			# imBB = ImageDraw.Draw(image)
			# # imBB.ellipse(bounds, fill=None, outline ="blue")
			# imBB.rectangle(bounds, fill=None, outline="red")

			# image.show()

			# image = Image.open(imageFile)

			# imCrop = image.crop(bounds)
			# imCrop.show()



## If file is run directly
if __name__ == '__main__':

	# Input directory
	inDir = 'input/309_tm_25C_top_need_35C/'
	# inDir = 'input/309_tm_26C_bot_need_40C/'

	# Run main function
	main(inDir)