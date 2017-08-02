import math

def find(x_dim, y_dim, n):
	
	# Round number_of_splits to the nearest square number
	base = round(math.sqrt(n))
	n = base*base

	# Size of subimages. MUST BE EQUAL TO VALUE IN WATERSHED.PY
	subimage_size = 500	

	# Make sure each split is big enough for at least 2 subimages in each dimension. If not, there's
	# really little reason to be splitting this image in the first place. 
	while x_dim/base < subimage_size*2 or y_dim/base < subimage_size*2:
		base -= 1
		print "Too many splits chosen, reducing to: %i" %(base*base)
	
	# Creates dimensions for a square based on the number of splits requested.
	# Integer truncation rounds down to the nearest whole number
	WIDTH = int(base)
	HEIGHT = WIDTH
	print "Input Dimensions: %i x %i" %(x_dim, y_dim)
	cols = int(x_dim / 100)*100
	rows = int(y_dim / 100)*100
	print "Image Dimensions: %i x %i" %(cols, rows)
	x_size = int(cols / WIDTH)
	y_size = int(rows / HEIGHT)
	print "Split Dimensions: %i x %i" %(x_size, y_size)

	factors_x = factor(x_size)
	factors_y = factor(y_size)
	for i in factors_x:
		if i > 400:
			block_x = i
			break
	for i in factors_y:
		if i > 400:
			block_y = i
			break

	# block_x = float(x_size)/int(x_size/500)
	# block_y = float(y_size)/int(y_size/500)

	print "Block Size: %f x %f" %(block_x, block_y)

	print "Number of x_blocks: %f" %(float(x_size)/block_x)
	print "Number of y_blocks: %f" %(float(y_size)/block_y)

# Returns a sorted list of all of the factors of number
# using trial division.
#	http://www.calculatorsoup.com/calculators/math/factors.php
def factor(number):

	factors = []
	s = int(math.ceil(math.sqrt(number)))

	for i in range(1,s):
		c = float(number) / i
		if int(c) == c:
			factors.append(c)
			factors.append(number/c)

	factors.sort()

	return factors
