import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import sys
from PIL import Image
from absl import flags, app
import numpy as np
import tensorflow as tf
import pdb
import functools
import matplotlib.pyplot as plt
import scipy
import scipy.misc
flags.DEFINE_string('data_dir', 'hw2_data',
										'Directory containing sub-directories tf_segmentation and '
										'images.')
FLAGS = flags.FLAGS
num_classes = 21
image_size = 128

def get_filename_data_readers(image_ids_file, get_labels=False,
															x_jpg_dir=None, y_png_dir=None):
	"""Given image IDs file, returns Datasets, which generate image paths.

	The goal of this function is to convert from image IDs to image paths.
	Specifically, the return type should be:
		if get_labels == False, return type should be tf.data.Dataset.
		Otherwise, return type should be pair (tf.data.Dataset, tf.data.Dataset).
	In both cases, the Dataset objects should be "not batched".

	For example, if the file contains 2 lines: "0000\n0001", then the returned
	dataset should give an iterator that when its tensor is ran, gives "0000" the
	first time and gives "0001" the second time.

	Args:
		image_ids_file: text with one image ID per line.
		get_labels: If set, returns 2 Datasets: the containing the image files (x)
			and the second containing the segmentation labels (y). If not, returns
			only the first argument.
		x_jpg_dir: Directory where each image lives. Specifically, image with
			ID "image1" will live on "x_jpg_dir/image1.jpg".
		y_png_dir: Directory where each segmentation mask lives. Specifically,
			image with ID "image1" will live on "x_png_dir/image1.png".

	Returns:
		instance of tf.data.Dataset, or pair of instances (if get_labels == True).
	"""
	x_jpg_dir = x_jpg_dir or os.path.join(FLAGS.data_dir, 'images')
	y_png_dir = y_png_dir or os.path.join(FLAGS.data_dir, 'tf_segmentation')
	# TODO(student): Write code.

	lines_dataset =  tf.data.TextLineDataset(image_ids_file)
	images_dataset = lines_dataset.map(lambda x: x_jpg_dir + os.path.sep + x + '.jpg')
	if get_labels:
		labels_dataset = lines_dataset.map(lambda x: y_png_dir + os.path.sep + x + '.png')
		return images_dataset, labels_dataset
	else:
		return images_dataset

def decode_image_with_padding(im_file, decode_fn=tf.image.decode_image,
															channels=3, pad_upto=500):
	"""Reads an image, decodes, and pads its spatial dimensions, all in TensorFlow

	Args:
		im_file: tf.string tensor, containing path to image file.
		decode_fn: Tensorflow function for converting
		channels: Image channels to decode. For data (x), set to 3 channels (i.e. RGB).
			For labels (segmentation masks), set to 1, because other 2 channels contain
			identical information.
		pad_upto: Number of pixels to pad to.

	Returns:
		Pair of Tensors:
			The first must be tf.int vector with 2 entries: containing the original height
				and width of the image.
			The second must be a tf.int matrix with size (pad_upto, pad_upto, 3)
				i.e. the contents of the image, with zero-padding.
	"""
	# TODO(student): Write code.
	content = tf.io.read_file(im_file)
	raw_img = functools.partial(decode_fn, channels=channels)(content)
	raw_shape = tf.shape(raw_img)
	paddings = [[0, pad_upto-raw_shape[0]], [0, pad_upto-raw_shape[1]], [0, 0]]

	return raw_shape[:2], tf.pad(raw_img, paddings)



def make_loss_mask(shapes):
	"""Given tf.int Tensor matrix with shape [N, 2], make N 2D binary masks.

	These binary masks will be used "to mask the loss". Specifically, if the
	image is shaped as (300 x 400) and therefore so its labels, we only want
	to penalize the model for misclassifying within the image boundary (300 x 400)
	and ignore values outside (e.g. at pixel [350, 380]).

	Args:
		shapes: tf.int Tensor with shape [N, 2]. Entry shapes[i] will be a vector:
			[image height, image width].

	Returns:
		tf.float32 mask of shape [N, 500, 500], with mask[i, h, w] set to 1.0
		iff shapes[i, 0] < h and shapes[i, 1] < w.
	"""
	# TODO(student): Write code.

	masks = tf.sequence_mask(shapes, 500)
	masks.set_shape([None, 2, 500])
	n_sample = tf.shape(shapes)[0]

	v1 = tf.cast(tf.broadcast_to(tf.expand_dims(masks[:, 0, :], 2), (n_sample, 500, 500)), tf.float32)
	v2 = tf.cast(tf.broadcast_to(tf.expand_dims(masks[:, 1, :], 2), (n_sample, 500, 500)), tf.float32)
	return v1*tf.transpose(v2, [0, 2, 1])



def read_image_pair_with_padding(x_im_file, y_im_file, pad_upto=500):
	"""Reads image pair (image & segmentation). You might find it useful.

	It only works properly, if you implemented `decode_image_with_padding`. If you
	do not find this function useful, ignore it.
	not have to use this function, if you do not find it useful.

	Args:
		x_im_file: Full path to jpg image to be segmented.
		y_im_file: Full path to png image, containing ground-truth segmentation.
		pad_upto: The padding of the images.

	Returns:
		tuple of tensors with 3 entries:
			int tensor of 2-dimensions.
	"""
	shape, im_x = decode_image_with_padding(x_im_file)
	_, im_y = decode_image_with_padding(y_im_file, tf.image.decode_png, channels=1)
	im_x = tf.image.resize(im_x, [image_size, image_size]) / 255.0
	im_y = tf.cast(tf.image.resize(tf.clip_by_value(tf.cast(im_y, tf.int32), 0, num_classes) % num_classes, [image_size, image_size]), tf.int32)

	#TODO: data augmentation
	return im_x, im_y

def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
	"""Upsamples an input.
	Conv2DTranspose => Batchnorm => Dropout => Relu
	Args:
		filters: number of filters
		size: filter size
		norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
		apply_dropout: If True, adds the dropout layer
	Returns:
		Upsample Sequential Model
	"""

	initializer = tf.random_normal_initializer(0., 0.02)

	result = tf.keras.Sequential()
	result.add(
			tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
																			padding='same',
																			kernel_initializer=initializer,
																			use_bias=False))

	if norm_type.lower() == 'batchnorm':
		result.add(tf.keras.layers.BatchNormalization())
	elif norm_type.lower() == 'instancenorm':
		result.add(InstanceNormalization())

	if apply_dropout:
		result.add(tf.keras.layers.Dropout(0.5))

	result.add(tf.keras.layers.ReLU())

	return result

class SegmentationModel:
	"""Class that can segment images into 21 classes (class 0 being background).

	You must implement the following in this class:
	+ load()
	+ predict()
	which will be called by the auto-grader

	+ train()
	+ save()
	which will NOT be called by the auto-grader, but you must implement them for
	repeatability. After the grading period, we will re-train all models to make
	sure their present training will get their results.
	"""

	def __init__(self):
		# 0: background class, {1, .., 20} are for class labels.
		self.num_classes = 21
		self.sess = tf.keras.backend.get_session()  # You can change or remove
		self.batch_size = 64  # You can change or remove
		self.smallest_weight = None
		if os.path.exists('output_precomputed.npz'):
			self.output_precomputed = np.load('output_precomputed.npz')
		self.image_size = 128

	def load(self, model_in_file):
		"""Restores parameters of model from `model_in_file`."""
		self.build_model()

		npz_data = np.load(model_in_file)
		values_dict = {f: npz_data[f] for f in npz_data.files}
		ops = []
		for v in tf.global_variables():
			if v.name in values_dict:
				ops.append(v.assign(values_dict[v.name]))

		pdb.set_trace()
		#ops = [v.assign(values_dict[v.name]) for v in tf.global_variables()]
		self.sess.run(ops)

	def _save_weight(self):
		tf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		self.smallest_weight = self.sess.run(tf_vars)

	def _load_weights(self):
		tf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
		ops = []
		for i_tf in range(len(self.smallest_weight)):
			ops.append(tf.assign(tf_vars[i_tf], self.smallest_weight[i_tf]))
		self.sess.run(ops)

	def predict(self, images):
		"""Predicts segments of some images.

		This method WILL BE CALLED by auto-grader. Please do not change signature of
		function [though adding optional arguments is fine].

		Args:
			images: List of images. All will be padded to 500x500 by autograder. The
				list can be a primitive list or a numpy array. To go from the former to
				the latter, just wrap as np.array(images).

		Returns:
			List of predictions (or numpy array). If np array, must be of shape:
			N x 500 x 500 x 21, where N == len(images). Our evaluation script will
			take the argmax on the last column for accuracy (and might calculate AUC).
		"""
		#testing_dataset = tf.data.Dataset.from_tensor_slices(images)
		#tf.keras.backend.set_learning_phase(0)
		testing_dataset = tf.data.Dataset.from_tensor_slices(np.asarray(images)).map(lambda x: tf.image.resize(x, [self.image_size, self.image_size]) / 255.0)
		#testing_dataset_shape = tf.data.Dataset.from_tensor_slices(np.full((len(images), 2), 500, dtype=np.int32))
		testing_iterator_X = tf.data.Dataset.zip((testing_dataset, )).batch(self.batch_size).make_initializable_iterator()

		self.sess.run(testing_iterator_X.initializer)
		testing_handle_X = self.sess.run(testing_iterator_X.string_handle())

		final_output = np.zeros([len(images), 500, 500, num_classes])
		j = 0
		count = 0
		while True:
			try:
				[test_output] = self.sess.run(
					[self.output],
						feed_dict={
							self.is_training: False,
							self.handle_X: testing_handle_X,
						}
				)
				this_len = len(test_output)
				for z in range(len(test_output)):
					for dim in range(num_classes):
						final_output[count+z:count+z+1, :, :, dim] = scipy.misc.imresize(test_output[z, :, :, dim], [500, 500])

				#final_output[count:count+this_len, :, :, :] = test_output
				to = final_output[count:count+this_len, :, :, :].argmax(axis=-1)
				'''
				pdb.set_trace()
				for z in range(this_len):
					plt.matshow(to[z])
					plt.colorbar()
					plt.show()
				'''
				count += this_len
				print(f'Batch: {j}')
				j += 1
			except tf.errors.OutOfRangeError:
				break
		return final_output


	def train(self, train_ids_file):
		"""Trains the model.

		This method WILL BE CALLED by our scripts, after submission period. Please
		do not add required arguments. Feel free to completely erase its body.

		Args:
			train_ids_file: file containing image IDs.
		"""
		# TODO(student): Feel free to remove if you do not use.
		#tf.keras.backend.set_learning_phase(1)
		file_pairs = tf.data.Dataset.zip(get_filename_data_readers(train_ids_file, True))
		#training_dataset = file_pairs.shuffle(buffer_size=2500).map(read_image_pair_with_padding).batch(self.batch_size)
		training_dataset = file_pairs.map(read_image_pair_with_padding).batch(self.batch_size)
		training_dataset_X = training_dataset.map(lambda a, b: a)
		training_dataset_Y = training_dataset.map(lambda a, b: b)

		training_iterator_X = training_dataset_X.make_initializable_iterator()
		training_iterator_Y = training_dataset_Y.make_initializable_iterator()

		for i in range(5):
			self.sess.run(training_iterator_X.initializer)
			self.sess.run(training_iterator_Y.initializer)
			training_handle_X = self.sess.run(training_iterator_X.string_handle())
			training_handle_Y = self.sess.run(training_iterator_Y.string_handle())
			j = 0
			loss_ary = []
			while True:
				try:
					[train_loss, train_step, train_pred] = self.sess.run(
						[self.loss, self.train_op, self.pred],
							feed_dict={
								self.is_training: True,
								self.handle_X: training_handle_X,
								self.handle_Y: training_handle_Y,
							}
					)
					if j == 0:
						plt.imshow(train_pred[0])
						plt.colorbar()
						plt.show()
						pdb.set_trace()
					loss_ary.append(train_loss)
					j += 1
					print('Epoch', i, 'Batch', j, train_loss)
				except tf.errors.OutOfRangeError:
					break
			self.sess.run(self.lr_decay_op)
			print('Apply lr decay, new lr: %f' % self.sess.run(self.lr))
			print(f'Epoch: {i}, Avg Loss: {np.mean(loss_ary)}')
			self.save("model_file_no_{}.pickle".format(i))


		print('Done training')

	def test(self, test_ids_file):
		#tf.keras.backend.set_learning_phase(0)

		files = get_filename_data_readers(test_ids_file, False)
		testing_dataset = files.map(decode_image_with_padding).batch(self.batch_size)
		testing_iterator_X = testing_dataset.make_initializable_iterator()
		self.sess.run(testing_iterator_X.initializer)
		testing_handle_X = self.sess.run(testing_iterator_X.string_handle())
		j = 0
		count = 0
		final_output =[]

		while True:
			try:
				[test_pred] = self.sess.run(
					[self.pred],
						feed_dict={
							self.is_training: False,
							self.handle_X: testing_handle_X,
						}
				)

				for mat in test_pred:
					final_output.append(mat)
					img = Image.fromarray(mat.astype(np.uint8))
					filename = 'pred/{:04d}.png'.format(count)
					count += 1
					img.save(filename)
				print(f'Batch: {j}')
				j += 1
			except tf.errors.OutOfRangeError:
				break

		final_output = np.asarray(final_output).astype(np.int32)
		np.savez(open('output_precomputed.npz', 'wb'), final_output)

	def save(self, model_out_file):
		"""Saves model parameters to disk."""
		variables_dict = {v.name: v for v in tf.global_variables()}
		values_dict = self.sess.run(variables_dict)
		np.savez(open(model_out_file, 'wb'), **values_dict)

	# TODO(student): Feel free to remove if you do not use.
	def build_model(self):
		handle_X = tf.placeholder(tf.string, shape=[])
		handle_Y = tf.placeholder(tf.string, shape=[])

		iterator_X = tf.data.Iterator.from_string_handle(handle_X, (tf.float32, ), (tf.TensorShape([None, self.image_size, self.image_size, 3]), ))
		iterator_Y = tf.data.Iterator.from_string_handle(handle_Y, (tf.int32, ), (tf.TensorShape([None, self.image_size, self.image_size, 1]), ))
		[X] = iterator_X.get_next()
		[Y] = iterator_Y.get_next()

		#masks = tf.image.resize_bilinear(tf.expand_dims(make_loss_mask(shapes), 3), [self.image_size, self.image_size], align_corners=True)
		is_training = tf.placeholder_with_default(False, shape=(), name='is_training')
		keep_prob = tf.cond(is_training, lambda:tf.constant(0.85), lambda:tf.constant(1.0))

		# hack for keras
		# save weights and reinitalize
		base_model = tf.keras.applications.MobileNetV2(input_shape=[image_size, image_size, 3], include_top=False)
		n_pre_train = len(tf.trainable_variables())
		self._save_weight()

		layer_names = [
				'block_1_expand_relu',   # 64x64
				'block_3_expand_relu',   # 32x32
				'block_6_expand_relu',   # 16x16
				'block_13_expand_relu',  # 8x8
				'block_16_project',      # 4x4
		]
		layers = [base_model.get_layer(name).output for name in layer_names]

		down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
		down_stack.trainable = False

		up_stack = [
				upsample(512, 3),  # 4x4 -> 8x8
				upsample(256, 3),  # 8x8 -> 16x16
				upsample(128, 3),  # 16x16 -> 32x32
				upsample(64, 3),   # 32x32 -> 64x64
		]

		last = tf.keras.layers.Conv2DTranspose(
				self.num_classes, 3, strides=2,
				padding='same', activation='softmax')  #64x64 -> 128x128

		# Downsampling through the model
		skips = down_stack(X)
		x = skips[-1]
		skips = reversed(skips[:-1])

		# Upsampling and establishing the skip connections
		for up, skip in zip(up_stack, skips):
			x = up(x)
			concat = tf.keras.layers.Concatenate()
			x = concat([x, skip])

		x = last(x)

		#filters_1 = tf.get_variable('f1', shape=(1, 1, 512, self.num_classes))
		#filters_2 = get_bilinear_filter('d1', (32, 32, self.num_classes, self.num_classes), 32)
		lr = tf.Variable(0.001, trainable=False)
		optimizer = tf.train.AdamOptimizer(lr)
		output_final = x
		# 255 is clamped to 0
		Y = tf.reshape(Y, [-1, self.image_size, self.image_size])
		Y_bin = tf.clip_by_value(Y, 0, 1)
		#masks = masks * Y_bin
		#masks = Y_bin
		loss = tf.reduce_mean(tf.boolean_mask(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=output_final), Y_bin))
		#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=output_final))
		#loss = tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(Y, output_final), Y_bin)
		pred = tf.argmax(output_final, axis=3)

		lr_decay_op = lr.assign(lr * 0.99)

		update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
		with tf.control_dependencies(update_ops):
			train_op = optimizer.minimize(loss, var_list=tf.trainable_variables()[n_pre_train:])
		#pdb.set_trace()

		# hack for keras.
		# save weights and reinitalize
		self.sess.run(tf.global_variables_initializer())
		self._load_weights()

		self.loss = loss
		self.pred = pred
		self.output = output_final
		self.train_op = train_op
		self.is_training = is_training
		self.handle_X = handle_X
		self.handle_Y = handle_Y
		self.lr_decay_op = lr_decay_op
		self.lr = lr


class Evaluator:
	def __init__(self, sess=None, test_dir='hw2_test_data'):
		self.sess = sess or tf.Session()
		filename = tf.placeholder(tf.string, None)
		im = tf.image.decode_image(tf.io.read_file(filename))
		self.shapes = []
		self.images = []
		self.labels = []
		import glob
		png_mask_files = sorted(glob.glob(os.path.join(test_dir, 'tf_segmentation', '*.png')))
		for png_mask_file in png_mask_files:
			jpg_im_file = os.path.basename(png_mask_file).replace('.png', '.jpg')
			jpg_im_file = os.path.join(
						os.path.dirname(png_mask_file), '..', 'images', jpg_im_file)

			if not os.path.exists(jpg_im_file):
				print('Ignoring mask for non-existing image file: ' + jpg_im_file)
				continue

			image_x = self.sess.run(im, {filename: jpg_im_file})
			image_y = self.sess.run(im, {filename: png_mask_file})[:, :, 0]
			image_shape = image_x.shape[:2]  # memorize image (height, width)
			padded_x = np.zeros([500, 500, 3], dtype='uint8')
			padded_x[:image_shape[0], :image_shape[1]] = image_x

			self.shapes.append(image_shape)
			self.images.append(padded_x)
			self.labels.append(image_y)

	def evaluate(self, model, num_examples=1000):
		print('evaluating ...')
		batch_h = model.predict(self.images[:num_examples])
		batch_y = self.labels[:num_examples]
		batch_shape = self.shapes[:num_examples]
		accuracies = []
		for h, y, shape in zip(batch_h, batch_y, batch_shape):
			eval_at_y = (y > 0) * (y <= 20)
			top1 = h[:shape[0], :shape[1]].argmax(axis=-1)
			accuracy = np.sum((top1 == y) * eval_at_y)
			# as a percent:
			accuracy /= np.sum(eval_at_y)
			accuracies.append(accuracy)
		return np.mean(accuracies)


def main(argv):
	# TODO(student): Feel free to write whatever here. Our auto-grader will *not*pdb
	# invoke this at all.
	if not os.path.exists(FLAGS.data_dir):
		sys.stderr.write('Error: Data directory not found: %s\n' % FLAGS.data_dir)

	'''
	file_ids = get_filename_data_readers("id_file.txt", False)
	imgs = file_ids.map(functools.partial(decode_image_with_padding, channels=3)).batch(100)
	imgs_iter = imgs.make_initializable_iterator()

	sess = tf.Session()
	sess.run(imgs_iter.initializer)
	#batch = sess.run(imgs_iter.get_next())
	#shapes = batch[0]
	shapes = imgs_iter.get_next()[0]
	masks = make_loss_mask(shapes)

	#img_tensor = sess.run(imgs.make_one_shot_iterator().get_next())

	pdb.set_trace()
	'''
	model = SegmentationModel()
	#model.build_model()
	model.load("unet_keras.pickle")
	model.train("id_file.txt")
	#model.save("model_file_lr")
	#model.load("model_file_no_9.pickle")
	#model.train("id_file.txt")
	#model.test("id_file_submit.txt")

	ev = Evaluator(model.sess)
	acc = ev.evaluate(model)
	print('acc', acc)

	print('OK')
if __name__ == '__main__':
	app.run(main)
