import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
from IPython.display import clear_output
import matplotlib.pyplot as plt
import pdb
import IPython
import functools
data_dir = 'hw2_data'
image_size = 128
num_classes = 21
#tf.enable_eager_execution()

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
	x_jpg_dir = x_jpg_dir or os.path.join(data_dir, 'images')
	y_png_dir = y_png_dir or os.path.join(data_dir, 'tf_segmentation')
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

"""## Download the Oxford-IIIT Pets dataset

The dataset is already included in TensorFlow datasets, all that is needed to do is download it. The segmentation masks are included in version 3.0.0, which is why this particular version is used.
"""

#dataset, info = tfds.load('oxford_iiit_pet:3.0.0', with_info=True)

"""The following code performs a simple augmentation of flipping an image. In addition,  image is normalized to [0,1]. Finally, as mentioned above the pixels in the segmentation mask are labeled either {1, 2, 3}. For the sake of convinience, let's subtract 1 from the segmentation mask, resulting in labels that are : {0, 1, 2}."""

def normalize(input_image, input_mask):
	input_image = tf.cast(input_image, tf.float32) / 255.0
	input_mask -= 1
	return input_image, input_mask

def load_image_train(datapoint):
	input_image = tf.image.resize(datapoint['image'], (128, 128))
	input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))


	if np.random.uniform(()) > 0.5:
		input_image = tf.image.flip_left_right(input_image)
		input_mask = tf.image.flip_left_right(input_mask)

	input_image, input_mask = normalize(input_image, input_mask)

	return input_image, input_mask

def load_image_test(datapoint):
	input_image = tf.image.resize(datapoint['image'], (128, 128))
	input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

	input_image, input_mask = normalize(input_image, input_mask)

	return input_image, input_mask

"""The dataset already contains the required splits of test and train and so let's continue to use the same split."""

#TRAIN_LENGTH = info.splits['train'].num_examples
TRAIN_LENGTH = 2330
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

#train = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#test = dataset['test'].map(load_image_test)
training_filepairs = tf.data.Dataset.zip(get_filename_data_readers("id_file.txt", True))
train = training_filepairs.map(read_image_pair_with_padding)
validation_filepairs = tf.data.Dataset.zip(get_filename_data_readers("id_file_test.txt", True, x_jpg_dir=os.path.join('hw2_test_data', 'images'), y_png_dir=os.path.join('hw2_test_data', 'tf_segmentation')))
test = validation_filepairs.map(read_image_pair_with_padding)
#pdb.set_trace()
train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test.batch(BATCH_SIZE).repeat()

"""Let's take a look at an image example and it's correponding mask from the dataset."""

def display(display_list):
	plt.figure(figsize=(15, 15))

	title = ['Input Image', 'True Mask', 'Predicted Mask']
	if len(display_list) == 3:
		display_list[2] = sess.run(display_list[2])[:, :, 0]
	#IPython.embed()
	for i in range(len(display_list)):
		plt.subplot(1, len(display_list), i+1)
		plt.title(title[i])
		#pdb.set_trace()
		plt.imshow(display_list[i])
		if i >= 1:
			plt.colorbar()
		#plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
		plt.axis('off')
	plt.show()

sess = tf.keras.backend.get_session()
iterator = train.make_initializable_iterator()
sess.run(iterator.initializer)
next_item = iterator.get_next()

sample_image, sample_mask = sess.run(next_item)
display([sample_image, sample_mask[:, :, 0]])
sess.run(iterator.initializer)

"""## Define the model
The model being used here is a modified U-Net. A U-Net consists of an encoder (downsampler) and decoder (upsampler). In-order to learn robust features, and reduce the number of trainable parameters, a pretrained model can be used as the encoder. Thus, the encoder for this task will be a pretrained MobileNetV2 model, whose intermediate outputs will be used, and the decoder will be the upsample block already implemented in TensorFlow Examples in the [Pix2pix tutorial](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/pix2pix/pix2pix.py).

The reason to output three channels is because there are three possible labels for each pixel. Think of this as multi-classification where each pixel is being classified into three classes.
"""

OUTPUT_CHANNELS = num_classes

"""As mentioned, the encoder will be a pretrained MobileNetV2 model which is prepared and ready to use in [tf.keras.applications](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications). The encoder consists of specific outputs from intermediate layers in the model. Note that the encoder will not be trained during the training process."""

base_model = tf.keras.applications.MobileNetV2(input_shape=[image_size, image_size, 3], include_top=False)

# Use the activations of these layers
layer_names = [
		'block_1_expand_relu',   # 64x64
		'block_3_expand_relu',   # 32x32
		'block_6_expand_relu',   # 16x16
		'block_13_expand_relu',  # 8x8
		'block_16_project',      # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

"""The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples."""

up_stack = [
		upsample(512, 3),  # 4x4 -> 8x8
		upsample(256, 3),  # 8x8 -> 16x16
		upsample(128, 3),  # 16x16 -> 32x32
		upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels):

	# This is the last layer of the model
	last = tf.keras.layers.Conv2DTranspose(
			output_channels, 3, strides=2,
			padding='same', activation='softmax')  #64x64 -> 128x128

	inputs = tf.keras.layers.Input(shape=[128, 128, 3])
	x = inputs

	# Downsampling through the model
	skips = down_stack(x)
	x = skips[-1]
	skips = reversed(skips[:-1])

	# Upsampling and establishing the skip connections
	for up, skip in zip(up_stack, skips):
		x = up(x)
		concat = tf.keras.layers.Concatenate()
		x = concat([x, skip])

	x = last(x)

	return tf.keras.Model(inputs=inputs, outputs=x)

"""## Train the model
Now, all that is left to do is to compile and train the model. The loss being used here is losses.sparse_categorical_crossentropy. The reason to use this loss function is because the network is trying to assign each pixel a label, just like multi-class prediction. In the true segmentation mask, each pixel has either a {0,1,2}. The network here is outputting three channels. Essentially, each channel is trying to learn to predict a class, and losses.sparse_categorical_crossentropy is the recommended loss for such a scenario. Using the output of the network, the label assigned to the pixel is the channel with the highest value. This is what the create_mask function is doing.
"""

def mean_pred(y_true, y_pred):
	Y_bin = tf.clip_by_value(y_true, 0, 1)[:, :, :, 0]
	#pdb.set_trace()
	loss = tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred), Y_bin)
	return loss

model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss=mean_pred,
							metrics=['accuracy'])

"""Let's try out the model to see what it predicts before training."""

def create_mask(pred_mask):
	pred_mask = tf.argmax(pred_mask, axis=-1)
	pred_mask = pred_mask[..., tf.newaxis]
	return pred_mask[0]

def show_predictions(dataset=None, num=1):
	if dataset:
		for image, mask in dataset.take(num):
			pred_mask = model.predict(image)
			display([image[0], mask[0][:, :, 0], create_mask(pred_mask)])
	else:
		display([sample_image, sample_mask[:, :, 0],create_mask(model.predict(sample_image[tf.newaxis, ...]))])

show_predictions()

"""Let's observe how the model improves while it is training. To accomplish this task, a callback function is defined below."""

class DisplayCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		clear_output(wait=True)
		show_predictions()
		print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

EPOCHS = 5
VAL_SUBSPLITS = 5
#VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS
VALIDATION_STEPS = 145//BATCH_SIZE//VAL_SUBSPLITS

def save(model_out_file):
	"""Saves model parameters to disk."""
	variables_dict = {v.name: v for v in tf.global_variables()}
	values_dict = sess.run(variables_dict)
	np.savez(open(model_out_file, 'wb'), **values_dict)

pdb.set_trace()
model_history = model.fit(train_dataset, epochs=1,
													steps_per_epoch=1,
													validation_steps=1,
													validation_data=test_dataset,
													callbacks=[DisplayCallback()])

pdb.set_trace()
save('unet_1.pickle')
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']

epochs = range(EPOCHS)

plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'bo', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss Value')
plt.ylim([0, 1])
plt.legend()
plt.show()

"""## Make predictions

Let's make some predictions. In the interest of saving time, the number of epochs was kept small, but you may set this higher to achieve more accurate results.
"""

show_predictions(test_dataset, 3)
pdb.set_trace()
print('Done')
"""## Next steps
Now that you have an understanding of what image segmentation is and how it works, you can try this tutorial out with different intermediate layer outputs, or even different pretrained model. You may also challenge yourself by trying out the [Carvana](https://www.kaggle.com/c/carvana-image-masking-challenge/overview) image masking challenge hosted on Kaggle.

You may also want to see the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) for another model you can retrain on your own data.
"""
