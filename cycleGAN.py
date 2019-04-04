from datetime import datetime
import os, random, time
import numpy as np
import tensorflow as tf
from scipy.misc import imsave

# parameters

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
EPOCHS = 200
BATCH_SIZE = 1
POOL_SIZE = 50 # number of generated images kept in training
NIDF = 64 # number of initial discriminator filters
NIGF = 64 # number of initial generator filters
ID_LOSS = False # add identity loss
SKIP = False # use skip connection over entire generator
DO = 'train' # train or test
SAVE_IMGS = True # save some results during training
SAVE_MODEL = True # save model after training

# data

DATASET_SIZES = {'monet2photo_train': 1072, 'monet2photo_test': 751,
                 'cezanne2photo_train': 525, 'cezanne2photo_test': 751,
                 'ukiyoe2photo_train': 562, 'ukiyoe2photo_test': 751,
                 'vangogh2photo_train': 400, 'vangogh2photo_test': 751,
                 'horse2zebra_train': 1334, 'horse2zebra_test': 140,
                 'summer2winter_yosemite_train': 1231, 'summer2winter_yosemite_test': 309,
                 'apple2orange_train' : 1019, 'apple2orange_test': 266}

def load_data(dataset_name, size_before_crop, do_shuffle = True, do_flipping = False):

	def preprocess(image):
		image = tf.image.resize_images(image, [size_before_crop, size_before_crop])
		image = tf.image.random_flip_left_right(image) if do_flipping else image
		image = tf.random_crop(image, [IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS])
		image = tf.subtract(tf.div(image, 127.5), 1)
		return image
	
	def parse_function(line):
		imagename_A, imagename_B = tf.decode_csv(line, record_defaults = rd)
		imagecontent_A, imagecontent_B = tf.read_file(imagename_A), tf.read_file(imagename_B)
		image_A = tf.image.decode_jpeg(imagecontent_A, channels = IMG_CHANNELS)
		image_B = tf.image.decode_jpeg(imagecontent_B, channels = IMG_CHANNELS)
		return preprocess(image_A), preprocess(image_B)

	csv_name = 'input/' + dataset_name[:dataset_name.rfind('_')] + '/' + dataset_name + '.csv'
	rd = [tf.constant([], dtype = tf.string), tf.constant([], dtype = tf.string)]
	dataset = tf.data.TextLineDataset([csv_name])
	dataset = dataset.map(parse_function)
	dataset = dataset.shuffle(1000) if do_shuffle else dataset # if test don't shuffle and don't repeat
	dataset = dataset.batch(BATCH_SIZE)
	dataset = dataset.repeat(EPOCHS + 10) if do_shuffle else dataset # if test don't shuffle and don't repeat
	iterator = dataset.make_one_shot_iterator()
	next_batch = iterator.get_next()
	return next_batch

# losses

def cycle_loss(real_images, fake_images):
    return tf.reduce_mean(tf.abs(real_images - fake_images))

def gan_loss_generator(prob_fake_is_real):
    return tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 1))

def gan_loss_discriminator(prob_real_is_real, prob_fake_is_real):
	first_term = tf.reduce_mean(tf.squared_difference(prob_real_is_real, 1))
	second_term = tf.reduce_mean(tf.squared_difference(prob_fake_is_real, 0)) 
	return 0.5 * (first_term + second_term)

def identity_loss(real_images, transformed_images):
	return tf.reduce_mean(tf.abs(real_images - transformed_images))

# layers

def leaky_relu(x, leak = 0.2):
	with tf.variable_scope("leaky_relu"):
		return tf.maximum(x, leak * x)

def instance_norm(x):
	with tf.variable_scope("instance_norm"):
		epsilon = 1e-5
		mean, var = tf.nn.moments(x, [1, 2], keep_dims = True)
		offset = tf.get_variable('offset', [x.get_shape()[-1]], initializer = tf.constant_initializer(0.0))
		scale = tf.get_variable('scale', [x.get_shape()[-1]], 
                                initializer = tf.truncated_normal_initializer(mean = 1.0, stddev = 0.02))
		return scale * tf.div(x - mean, tf.sqrt(var + epsilon)) + offset

def conv2d(input_conv, o_d = 64, f_h = 7, f_w = 7, s_h = 1, s_w = 1, stddev = 0.02, padding = "VALID", 
           name = "conv2d", do_norm = True, do_relu = True, relu_factor = 0):
	with tf.variable_scope(name):
		conv = tf.contrib.layers.conv2d(input_conv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn = None, 
                                        weights_initializer = tf.truncated_normal_initializer(stddev = stddev),
                                        biases_initializer = tf.constant_initializer(0.0))
		if do_norm:
			conv = instance_norm(conv)
		if do_relu:
			conv = tf.nn.relu(conv, "relu") if relu_factor == 0 else leaky_relu(conv, relu_factor)
		return conv

def deconv2d(input_conv, o_d = 64, f_h = 7, f_w = 7, s_h = 1, s_w = 1, stddev = 0.02, padding = "VALID", 
             name = "deconv2d", do_norm = True, do_relu = True, relu_factor = 0):
	with tf.variable_scope(name):
		deconv = tf.contrib.layers.conv2d_transpose(input_conv, o_d, [f_h, f_w], [s_h, s_w], padding, activation_fn = None, 
                                                    weights_initializer = tf.truncated_normal_initializer(stddev = stddev),
                                                    biases_initializer = tf.constant_initializer(0.0))
		if do_norm:
			deconv = instance_norm(deconv)
		if do_relu:
			deconv = tf.nn.relu(deconv, "relu") if relu_factor == 0 else leaky_relu(deconv, relu_factor)
		return deconv

# blocks

def resnet_block(input_res, dim, name = "resnet", padding = "REFLECT", res_relu = True): # 2 conv2d with a skip connection
    with tf.variable_scope(name):
        out_res = tf.pad(input_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c1")
        out_res = tf.pad(out_res, [[0, 0], [1, 1], [1, 1], [0, 0]], padding)
        out_res = conv2d(out_res, dim, 3, 3, 1, 1, 0.02, "VALID", "c2", do_relu = False)
        out_res = tf.nn.relu(out_res + input_res) if res_relu else out_res + input_res
        return out_res

def generator(input_gen, name = "generator"): # 9 resnet blocks
	with tf.variable_scope(name):
		f = 7
		ks = 3
		padding = "REFLECT"

		input_pad = tf.pad(input_gen, [[0, 0], [ks, ks], [ks, ks], [0, 0]], padding)
		o_c1 = conv2d(input_pad, NIGF, f, f, 1, 1, 0.02, "VALID", "c1")
		o_c2 = conv2d(o_c1, NIGF * 2, ks, ks, 2, 2, 0.02, "SAME", "c2")
		o_c3 = conv2d(o_c2, NIGF * 4, ks, ks, 2, 2, 0.02, "SAME", "c3")

		o_r1 = resnet_block(o_c3, NIGF * 4, "r1", padding)
		o_r2 = resnet_block(o_r1, NIGF * 4, "r2", padding)
		o_r3 = resnet_block(o_r2, NIGF * 4, "r3", padding)
		o_r4 = resnet_block(o_r3, NIGF * 4, "r4", padding)
		o_r5 = resnet_block(o_r4, NIGF * 4, "r5", padding)
		o_r6 = resnet_block(o_r5, NIGF * 4, "r6", padding)
		o_r7 = resnet_block(o_r6, NIGF * 4, "r7", padding)
		o_r8 = resnet_block(o_r7, NIGF * 4, "r8", padding)
		o_r9 = resnet_block(o_r8, NIGF * 4, "r9", padding)

		o_c4 = deconv2d(o_r9, NIGF * 2, ks, ks, 2, 2, 0.02, "SAME", "c4")
		o_c5 = deconv2d(o_c4, NIGF, ks, ks, 2, 2, 0.02, "SAME", "c5")
		o_c6 = conv2d(o_c5, IMG_CHANNELS, f, f, 1, 1, 0.02, "SAME", "c6", do_norm = False, do_relu = False)

		out_gen = tf.nn.tanh(input_gen + o_c6, "t1") if SKIP else tf.nn.tanh(o_c6, "t1")
		return out_gen

def discriminator(input_disc, name = "discriminator"): # 5 conv2d
	with tf.variable_scope(name):
		f = 4
		o_c1 = conv2d(input_disc, NIDF, f, f, 2, 2, 0.02, "SAME", "c1", do_norm = False, relu_factor = 0.2)
		o_c2 = conv2d(o_c1, NIDF * 2, f, f, 2, 2, 0.02, "SAME", "c2", relu_factor = 0.2)
		o_c3 = conv2d(o_c2, NIDF * 4, f, f, 2, 2, 0.02, "SAME", "c3", relu_factor = 0.2)
		o_c4 = conv2d(o_c3, NIDF * 8, f, f, 1, 1, 0.02, "SAME", "c4", relu_factor = 0.2)
		o_c5 = conv2d(o_c4, 1, f, f, 1, 1, 0.02, "SAME", "c5", do_norm = False, do_relu = False)
		return o_c5

# model

class CycleGAN:

	def __init__(self, dataset_name, lambda_A = 10, lambda_B = 10, lambda_I = 5, base_lr = 0.0002, do_flipping = True):
		self._dataset_name = dataset_name
		self._lambda_A = lambda_A # weight of cycle loss A
		self._lambda_B = lambda_B # weight of cycle loss B
		self._lambda_I = lambda_I # weight of identity loss
		self._base_lr = base_lr # base learning rate
		self._num_images_to_save = 20 # number of examples to save after each epoch
		self._output_dir = os.path.join('output', datetime.now().strftime("%Y%m%d-%H%M%S"))
		self._images_dir = os.path.join(self._output_dir, 'images')
		self._checkpoint_dir = 'checkpoints'
		self._size_before_crop = 286 # for random cropping in preprocessing
		self._do_flipping = do_flipping # random flipping of images in preprocessing
		self.pool_fake_images_A = np.zeros((POOL_SIZE, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
		self.pool_fake_images_B = np.zeros((POOL_SIZE, BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
		self.create_output_folders()

	def create_output_folders(self):
		if not os.path.exists(self._output_dir):
			os.makedirs(self._output_dir)
		if not os.path.exists(self._images_dir):
			os.makedirs(self._images_dir)
		if not os.path.exists(self._checkpoint_dir):
			os.makedirs(self._checkpoint_dir)

	def create_model(self):
		self.input_A = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name = "input_A")
		self.input_B = tf.placeholder(tf.float32, [BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name = "input_B")

		self.fake_pool_A = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name = "fake_pool_A")
		self.fake_pool_B = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name = "fake_pool_B")

		self.global_step = tf.train.get_or_create_global_step()
		self.num_fake_inputs = 0
		self.learning_rate = tf.placeholder(tf.float32, shape = [], name = "lr")

		with tf.variable_scope("model") as scope:

			self.prob_real_A_is_real = discriminator(self.input_A, "d_A")
			self.prob_real_B_is_real = discriminator(self.input_B, "d_B")

			self.fake_images_B = generator(self.input_A, name = "g_A")
			self.fake_images_A = generator(self.input_B, name = "g_B")

			scope.reuse_variables()

			self.wrong_fake_B = generator(self.input_A, name = "g_B")
			self.wrong_fake_A = generator(self.input_B, name = "g_A")

			self.prob_fake_A_is_real = discriminator(self.fake_images_A, "d_A")
			self.prob_fake_B_is_real = discriminator(self.fake_images_B, "d_B")

			scope.reuse_variables()

			self.cycle_images_A = generator(self.fake_images_B, "g_B")
			self.cycle_images_B = generator(self.fake_images_A, "g_A")

			self.prob_fake_pool_A_is_real = discriminator(self.fake_pool_A, "d_A")
			self.prob_fake_pool_B_is_real = discriminator(self.fake_pool_B, "d_B")

	def compute_losses(self):
		cycle_loss_A = self._lambda_A * cycle_loss(self.input_A, self.cycle_images_A)
		cycle_loss_B = self._lambda_B * cycle_loss(self.input_B, self.cycle_images_B)
		gan_loss_A = gan_loss_generator(self.prob_fake_A_is_real)
		gan_loss_B = gan_loss_generator(self.prob_fake_B_is_real)
		identity_loss_A = self._lambda_I * identity_loss(self.wrong_fake_A, self.input_B)
		identity_loss_B = self._lambda_I * identity_loss(self.wrong_fake_B, self.input_A)
		g_loss_A = cycle_loss_A + cycle_loss_B + gan_loss_B + identity_loss_A if ID_LOSS else cycle_loss_A + cycle_loss_B + gan_loss_B
		g_loss_B = cycle_loss_B + cycle_loss_A + gan_loss_A + identity_loss_B if ID_LOSS else cycle_loss_B + cycle_loss_A + gan_loss_A
		d_loss_A = gan_loss_discriminator(self.prob_real_A_is_real, self.prob_fake_pool_A_is_real)
		d_loss_B = gan_loss_discriminator(self.prob_real_B_is_real, self.prob_fake_pool_B_is_real)


		optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1 = 0.5)
		self.model_vars = tf.trainable_variables()

		d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
		g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
		d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
		g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]

		self.d_A_trainer = optimizer.minimize(d_loss_A, var_list = d_A_vars)
		self.d_B_trainer = optimizer.minimize(d_loss_B, var_list = d_B_vars)
		self.g_A_trainer = optimizer.minimize(g_loss_A, var_list = g_A_vars)
		self.g_B_trainer = optimizer.minimize(g_loss_B, var_list = g_B_vars)

		for var in self.model_vars:
		    print(var.name)

		self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
		self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
		self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
		self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

	def train(self):
		self.inputs = load_data(self._dataset_name + '_train', self._size_before_crop, True, self._do_flipping)
		self.create_model()
		self.compute_losses()

		init = (tf.global_variables_initializer(), tf.local_variables_initializer())
		saver = tf.train.Saver()
		num_images = DATASET_SIZES[self._dataset_name + '_train']

		with tf.Session() as sess:
			sess.run(init)
			t1 = time.time()

			for epoch in range(sess.run(self.global_step), EPOCHS): # Training Loop
				t0 = time.time()
				print("In epoch", epoch)
				if epoch < 100: # learning rate decreases after 100 epochs
					curr_lr = self._base_lr
				else:
					curr_lr = self._base_lr - self._base_lr * (epoch - 100) / 100
				if SAVE_IMGS:
					self.save_images(sess, epoch)

				for i in range(0, num_images):
					print("Processing batch {}/{} in epoch {}".format(i, num_images, epoch))
					inputs_A, inputs_B = sess.run(self.inputs)

					# optimizing generator A
					_, fake_B_temp, summary_str = sess.run(fetches = [self.g_A_trainer, self.fake_images_B, self.g_A_loss_summ],
                                                           feed_dict = {self.input_A : inputs_A, self.input_B : inputs_B, 
                                                                        self.learning_rate : curr_lr})

					fake_B_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.pool_fake_images_B)

					# optimizing discriminator B
					_, summary_str = sess.run(fetches = [self.d_B_trainer, self.d_B_loss_summ],
                                              feed_dict = {self.input_A : inputs_A, self.input_B : inputs_B,
                                                           self.learning_rate : curr_lr, self.fake_pool_B : fake_B_temp1})

					# optimizing generator B
					_, fake_A_temp, summary_str = sess.run(fetches = [self.g_B_trainer, self.fake_images_A, self.g_B_loss_summ],
                                                           feed_dict = {self.input_A : inputs_A, self.input_B : inputs_B,
                                                                        self.learning_rate : curr_lr})

					fake_A_temp1 = self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.pool_fake_images_A)

					# optimizing discriminator A
					_, summary_str = sess.run(fetches = [self.d_A_trainer, self.d_A_loss_summ],
                                              feed_dict = {self.input_A : inputs_A, self.input_B : inputs_B,
                                                           self.learning_rate : curr_lr, self.fake_pool_A : fake_A_temp1})

					self.num_fake_inputs += 1

				sess.run(tf.assign(self.global_step, epoch + 1))
				print('--------- Epoch time:', time.time() - t0)

			print('Total time for', EPOCHS, 'epochs:', time.time() - t1)
			if SAVE_MODEL:
				print('Saving model...')
				saver.save(sess, os.path.join(self._checkpoint_dir, 'model.ckpt'))

	def save_images(self, sess, epoch): # save some results
		names = ['inputA_', 'inputB_', 'fakeA_', 'fakeB_', 'cycA_', 'cycB_']
		with open(os.path.join(self._output_dir, 'epoch_' + str(epoch) + '.html'), 'w') as v_html:
			for i in range(0, self._num_images_to_save):
				print("--- Saving image {}/{}".format(i, self._num_images_to_save))
				inputs_A, inputs_B = sess.run(self.inputs)
				fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp = sess.run(fetches = [self.fake_images_A, self.fake_images_B,
		            														           self.cycle_images_A, self.cycle_images_B], 
		            														feed_dict = {self.input_A : inputs_A,
		            																     self.input_B : inputs_B})
				tensors = [inputs_A, inputs_B, fake_B_temp, fake_A_temp, cyc_A_temp, cyc_B_temp]
				for name, tensor in zip(names, tensors):
					image_name = name + str(epoch) + "_" + str(i) + ".jpg"
					imsave(os.path.join(self._images_dir, image_name), ((tensor[0] + 1) * 127.5).astype(np.uint8))
					v_html.write("<img src=\"" + os.path.join('images', image_name) + "\">")
				v_html.write("<br>")

	def fake_image_pool(self, num_fakes, fake, fake_pool): # update pool of generated images
		if num_fakes < POOL_SIZE:
			fake_pool[num_fakes] = fake
			return fake
		else:
			p = random.random()
			if p > 0.5:
				random_id = random.randint(0, POOL_SIZE - 1)
				temp = fake_pool[random_id]
				fake_pool[random_id] = fake
				return temp
			else:
				return fake

	def test(self): # loads and tests a model
		print('--------- Test')
		self.inputs = load_data(self._dataset_name + '_test', self._size_before_crop, False, self._do_flipping)
		self.create_model()
		saver = tf.train.Saver()
		init = tf.global_variables_initializer()
		with tf.Session() as sess:
			sess.run(init)
			chkpt_fname = tf.train.latest_checkpoint(self._checkpoint_dir)
			saver.restore(sess, chkpt_fname)
			self._num_images_to_save = 100
			self.save_images(sess, 200)

# main

if __name__ == '__main__':
	dataset_name = 'vangogh2photo'
	cyclegan= CycleGAN(dataset_name)
	if DO == 'train':
		cyclegan.train()
	else:
		cyclegan.test()
	
