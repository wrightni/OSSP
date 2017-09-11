#title: Segment Feature Analysis
#author: Nick Wright

import numpy as np

def analyze_pan_image(input_image, watershed_image, entropy_image, date, segment_id=False):

	feature_matrix = []

	#If no segment id is provided, analyze the features for every watershed 
	# in the input image. If a segment id is provided, just analyze the features
	# for that one segment. 
	#We have to add +1 to num_ws because if the maximum value in watershed_image
	# is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1
	if segment_id == False:
		num_ws = int(np.amax(watershed_image)+1)
	else:
		num_ws = 1

	for ws in range(num_ws):

		if segment_id == False:
			current_sp = watershed_image==ws
		else:
			current_sp = watershed_image==segment_id

		#Skips any discontinuities in the watershed image
		if len(watershed_image[current_sp])==0:
			current_sp = input_image==0
	
		#Reset the feature list for each sample
		features = []

		ws_pixels = []

		ws_pixels = input_image[current_sp]

		#Average Pixel Value
		features.append(np.average(ws_pixels))
		#Pixel Median
		features.append(np.median(ws_pixels))
		#Segment Min
		features.append(np.amin(ws_pixels))
		#Segment Max
		features.append(np.amax(ws_pixels))
		#Standard Deviation
		features.append(np.std(ws_pixels))
		#Size of Superpixel 
		features.append(len(ws_pixels))

		#Adding the average entropy of the watershed to the feature matrix
		# entropy_image = entropy(bytescale(original_image), disk(4))
		entropy_of_segment = np.average(entropy_image[current_sp])

		features.append(entropy_of_segment)

		# Find the indices of the watershed pixels
		#	np.nonzero returns the indices of the elements that are non-zero
		sp_position = np.nonzero(current_sp)

		# Find the maximum and minimum of the x and y indicies. Add a 5 pixel
		# buffer in each dimension
		xMin = np.amin(sp_position[0])-5
		xMax = np.amax(sp_position[0])+5
		yMin = np.amin(sp_position[1])-5
		yMax = np.amax(sp_position[1])+5

		# Check if the min and max indicies are outside of the image
		# and if they are, set to the edge value. 
		if xMin<0:
			xMin=0
		if xMax>np.shape(input_image)[0]:
			xMax=np.shape(input_image)[0]
		if yMin<0:
			yMin=0
		if yMax>np.shape(input_image)[1]:
			yMax=np.shape(input_image)[1]

		# Define a region that includes the segment and the 5 pixel buffer
		region = input_image[xMin:xMax,yMin:yMax]
		entropy_region = entropy_image[xMin:xMax,yMin:yMax]
		region_exclude = ~current_sp[xMin:xMax,yMin:yMax]

		square_average = np.average(region[region_exclude])
		max_near = np.amax(region[region_exclude])
		std_near = np.std(region[region_exclude])
		entropy_near = np.average(entropy_region[region_exclude])

		features.append(square_average)
		features.append(std_near)
		features.append(max_near)	#This is the highest intensity neighboring pixel.
		features.append(entropy_near)

		# DATE
		features.append(date)
		
		# input_feature_matrix[i].append(int(input_filename[:4]))

		feature_matrix.append(features)
		
	return feature_matrix


def analyze_srgb_image(input_image, watershed_image, entropy_image, segment_id=False):

	feature_matrix = []

	#If no segment id is provided, analyze the features for every watershed 
	# in the input image. If a segment id is provided, just analyze the features
	# for that one segment. 
	#We have to add +1 to num_ws because if the maximum value in watershed_image
	# is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1
	if segment_id == False:
		num_ws = int(np.amax(watershed_image)+1)
	else:
		num_ws = 1

	for ws in range(num_ws):

		# Find the region of the image that matches the current
		# ws value
		if segment_id == False:
			current_sp = watershed_image==ws
		else:
			current_sp = watershed_image==segment_id

		# Skips any discontinuities in the watershed image
		if len(watershed_image[current_sp])==0:
			features = np.zeros(16)
			feature_matrix.append(features)
			continue

		# Reset the feature list for each sample
		features = []

		# Finds the pixel values for the current watershed (superpixel)
		# for all bands in the image
		ws_pixels_all = []
		for band in range(3):
			ws_pixels_single = input_image[:,:,band][current_sp]
			ws_pixels_all.append(ws_pixels_single)

		#Average Pixel Value in Superpixel
		for band in range(3):
			# Prevent division by zero. If the average is zero, we dont care what
			#	happens with this feature list, because the watershed doesnt
			# 	contain any image information, and therefore the classification will
			#	be overwritten later.
			if np.average(ws_pixels_all[band]) < 0.1:
				features.append(1)
				continue
			features.append(np.average(ws_pixels_all[band]))

		#Standard Deviation
		for band in range(3):
			features.append(np.std(ws_pixels_all[band]))

		# See Miao et al for band ratios
		#Band Ratio 1
		features.append((features[2]-features[0])/(features[2]+features[0]))
		#Band Ratio 2
		features.append((features[2]-features[1])/(features[2]+features[1]))
		#Band Ratio 3
		# Prevent division by 0
		if (2*features[2]-features[1]-features[0]) < 1:
			features.append(0)	
		else:
			features.append((features[1]-features[0])/(2*features[2]-features[1]-features[0]))	

		#Size of Superpixel 
		features.append(len(ws_pixels_all[0]))

		# Entropy
		entropy_of_segment = np.average(entropy_image[current_sp])
		features.append(entropy_of_segment)

		# Find the indices of the watershed pixels
		#	np.nonzero returns the indices of the elements that are non-zero
		sp_position = np.nonzero(current_sp)

		# Find the maximum and minimum of the x and y indicies. Add a 5 pixel
		# buffer in each dimension
		xMin = np.amin(sp_position[0])-5
		xMax = np.amax(sp_position[0])+5
		yMin = np.amin(sp_position[1])-5
		yMax = np.amax(sp_position[1])+5

		# Check if the min and max indicies are outside of the image
		# and if they are, set to the edge value.
		im_dim = np.shape(input_image)
		if xMin<0:
			xMin=0
		if xMax>im_dim[0]:
			xMax=im_dim[0]
		if yMin<0:
			yMin=0
		if yMax>im_dim[1]:
			yMax=im_dim[1]

		# Define a region that includes the segment and the 5 pixel buffer
		region = input_image[xMin:xMax,yMin:yMax]
		entropy_region = entropy_image[xMin:xMax,yMin:yMax]
		# Define 
		# current_sp[xMin:xMax,yMin:yMax] defines an area that contains the 
		#	super pixel and the border, where pixels within the SP are
		#	1s, and outside are 0s. The ~ operator flips these assignments.
		region_exclude = ~current_sp[xMin:xMax,yMin:yMax]

		# If region exclude has 0s for the entire region (happens when there is only one
		#	superpixel in the image), then set the outside values to be the same as the
		#	inside ones. I think this makes the valid assumption that if all of an image
		#	is the same, then the neighboring values are probably similar as well. 
		if np.max(region_exclude) == 0:
			square_average = features[0]
			max_near = np.amax(region)
			std_near = features[3]
			entropy_near = np.average(entropy_region)
		else:
			square_average = np.average(region[region_exclude])
			max_near = np.amax(region[region_exclude])
			std_near = np.std(region[region_exclude])
			entropy_near = np.average(entropy_region[region_exclude])

		features.append(square_average)
		features.append(std_near)
		features.append(max_near)	#This is the highest intensity neighboring pixel.
		features.append(entropy_near)

		# DATE
		features.append(0)
		# input_feature_matrix[i].append(int(input_filename[:4]))

		feature_matrix.append(features)

	return feature_matrix

#Calculates the features of the image we want to classify. These methods are a mirror
# of those used to create the training data set in "training_gui.py"
def analyze_ms_image(input_image, watershed_image, segment_id=False):

	feature_matrix = []
	
	#If no segment id is provided, analyze the features for every watershed 
	# in the input image. If a segment id is provided, just analyze the features
	# for that one segment. 
	#We have to add +1 to num_ws because if the maximum value in watershed_image
	# is 500, then there are 501 total watersheds Sum(0,1,...,499,500) = 500+1
	if segment_id == False:
		num_ws = int(np.amax(watershed_image)+1)
	else:
		num_ws = 1

	for ws in range(num_ws):
		
		# Find the region of the image that matches the current
		# ws value
		if segment_id == False:
			current_sp = watershed_image==ws
		else:
			current_sp = watershed_image==segment_id

		# Skips any discontinuities in the watershed image
		# Returns a zero array of length equal to the number of features
		if len(watershed_image[current_sp])==0:
			features = np.zeros(17)
			feature_matrix.append(features)
			continue

		# Reset the feature list for each sample
		features = []

		# Finds the pixel values for the current watershed (superpixel)
		# for all bands in the image
		ws_pixels_all = []
		for i in range(8):
			ws_pixels_single = input_image[:,:,i][current_sp]
			ws_pixels_all.append(ws_pixels_single)

		# Average Pixel Value in Superpixel for bands 1-7
		for band in range(7):
			# Prevent division by zero. If the average is zero, we dont care what
			#	happens with this feature list, because the watershed doesnt
			# 	contain any image information, and therefore the classification will
			#	be overwritten later.
			if np.average(ws_pixels_all[band]) == 0:
				features.append(1)
				continue
			features.append(np.average(ws_pixels_all[band]))


		# Important band ratios
		features.append(features[0]/features[2])
		features.append(features[1]/features[6])
		features.append(features[4]/features[6])
		features.append(features[3]/features[5])
		features.append(features[3]/features[6])
		features.append(features[3]/features[7])
		features.append(features[4]/features[6])

		# From now on, we are only working with the band 4? data
		input_image_b1 = input_image[:,:,3]

		# Find the indices of the watershed pixels
		#	np.nonzero returns the indices of the elements that are non-zero
		sp_position = np.nonzero(current_sp)

		# Find the maximum and minimum of the x and y indicies. Add a 5 pixel
		# buffer in each dimension
		xMin = np.amin(sp_position[0])-5
		xMax = np.amax(sp_position[0])+5
		yMin = np.amin(sp_position[1])-5
		yMax = np.amax(sp_position[1])+5

		# Check if the min and max indicies are outside of the image
		# and if they are, set to the edge value. 
		if xMin<0:
			xMin=0
		if xMax>np.shape(input_image_b1)[0]:
			xMax=np.shape(input_image_b1)[0]
		if yMin<0:
			yMin=0
		if yMax>np.shape(input_image_b1)[1]:
			yMax=np.shape(input_image_b1)[1]

		# Neighbor Average
		if np.amax(watershed_image) <= 1:
			square_average = features[0]
		else:
			# Define a region that includes the segment and the 5 pixel buffer
			region = input_image_b1[xMin:xMax,yMin:yMax]
			region_exclude = ~current_sp[xMin:xMax,yMin:yMax]

			square_average = np.average(region[region_exclude])
		
		features.append(square_average)

		# b1-b7 / b1+b7
		features.append((features[0]-features[6])/(features[0]+features[6]))
		# b3-b5 / b3+b5
		features.append((features[2]-features[4])/(features[2]+features[4]))

		feature_matrix.append(features)

	return feature_matrix
