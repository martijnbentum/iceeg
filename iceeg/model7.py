import tensorflow as tf


def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
												strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape,name = None):
	"""weight_variable generates a weight variable of a given shape."""
	print(shape)
	initial = tf.truncated_normal(shape, stddev=0.1)
	if type(name) != str: temp = tf.Variable(initial)
	else: temp = tf.Variable(initial,name = name)
	return temp


def bias_variable(shape,name = None):
	"""bias_variable generates a bias variable of a given shape."""
	initial = tf.constant(0.1, shape=shape)
	if type(name) != str: temp = tf.Variable(initial)
	else: temp = tf.Variable(initial,name = name)
	return temp

def next_batch(data,info, batch_size):
	nrows = data.shape[0]
	indices = random.sample(range(nrows),batch_size)
	return data[indices,:],info[indices,:]


def deepnn(x,nchannels,nsamples,fmap_size = 2400):
	"""deepnn builds the graph for a deep net for classifying digits.
	Args:
		x: an input tensor with the dimensions (N_examples, 784), where 784 is the
		number of pixels in a standard MNIST image.
	Returns:
		A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
		equal to the logits of classifying the digit into one of 10 classes (the
		digits 0-9). keep_prob is a scalar placeholder for the probability of
		dropout.
	"""
	# Reshape to use within a convolutional neural net.
	# Last dimension is for "features" - there is only one here, since images are
	# grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
	with tf.name_scope('reshape'):
		print(x)
		# x_image = tf.reshape(x, [-1, 24, 100, 1])
		x_image = tf.reshape(x, [-1, nchannels, nsamples, 1])
		print('reshaped')
		print(x_image)

	# First convolutional layer - maps one grayscale image to 32 feature maps.
	with tf.name_scope('conv1'):
		W_conv1 = weight_variable([1, 25, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		print(W_conv1,'W_conv1')
		print(b_conv1,'b_conv1')
		print(h_conv1,'h_conv1')


	# Second convolutional layer -- maps 32 feature maps to 64.
	with tf.name_scope('conv2'):
		W_conv2 = weight_variable([6, 1, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
		print(W_conv2,'W_conv2')
		print(b_conv2,'b_conv2')
		print(h_conv2,'h_conv2')

	# Pooling layer - downsamples by 2X.
	with tf.name_scope('pool1'):
		h_pool1 = max_pool_2x2(h_conv2)

	with tf.name_scope('conv3'):
		W_conv3 = weight_variable([5, 5, 64, 128])
		b_conv3 = bias_variable([128])
		h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)
		print(W_conv3,'W_conv3')
		print(b_conv3,'b_conv3')
		print(h_conv3,'h_conv3')

	# Second pooling layer.
	with tf.name_scope('pool2'):
		h_pool2 = max_pool_2x2(h_conv3)
		print(h_pool2,'h_pool2')


	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([8 * 25 * 128, fmap_size])
		b_fc1 = bias_variable([fmap_size])
		print(W_fc1,'W_fc1')
		print(b_fc1,'b_fc1')

		h_pool2_flat = tf.reshape(h_pool2, [-1,8 * 25* 128],name = 'h_pool3_flat-bla')
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1,name='h_fc1-bla')
		print(h_pool2_flat,'h_pool2_flat')
		print(h_fc1,'h_fc1')

	# Dropout - controls the complexity of the model, prevents co-adaptation of
	# features.
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32,name = 'keep_prob-bla')
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob,name = 'h_fc1_drop-bla')

	# Map the final features to 2 classes, one for each digit
	with tf.name_scope('fc2'):
		W_fc2 = weight_variable([fmap_size, 2],name='W_fc2-bla')
		b_fc2 = bias_variable([2],name='b_fc2-bla')
		print(W_fc2,'W_fc2')
		print(b_fc2,'b_fc2')

		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	print(y_conv,'y_conv')
	return y_conv, keep_prob

