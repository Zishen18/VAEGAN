import numpy as np
import tensorflow as tf
from deconv import deconv2d
import prettytensor as pt

def encoder(X, dim1, dim2, dim3, hidden_size):
	lay_end = (pt.wrap(X).reshape([batch_size, dim1, dim2, dim3]).conv2d(5,
			64, stride=2).conv2d(5, 128, stride=2).conv2d(5, 256,
			stride=2).flatten())
	z_mean = lay_end.fully_connected(hidden_size, activation_fn=None)
	z_log_sigma_sq = lay_end.fully_connected(hidden_size, activation_fn=None)
	return z_mean, z_log_sigma_sq


def generator(Z, batch_size, dim3):
	
	return (pt.wrap(Z).fully_connected(8*8*256).reshape([batch_size, 8, 8, 256]).deconv2d(5,
		256, stride=2).deconv2d(5, 128, stride=2).deconv2d(5, 32, stride=2).deconv2d(1, dim3,
		stride=1, activation_fn=tf.sigmoid).flatten())

def discriminator(D_I, dim1, dim2, dim3):
	
	descrim_conv = (pt.wrap(D_I).reshape([batch_size, dim1, dim2, dim3]).conv2d(5, 32, stride=1).
			conv2d(5, 128, stride=2).conv2d(5, 256, stride=2).conv2d(5, 256, stride=2).
			flatten())
	lth_layer = descrim_conv.fully_connected(1024, activation_fn=tf.nn.elu)
	D = lth_layer.fully_connected(1, activation_fn=tf.nn.sigmoid)
	return D, lth_layer

		
