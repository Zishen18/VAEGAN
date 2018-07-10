import numpy as np
import math


def create_image(img, dim1, dim2, dim3):
	img_reshape = np.reshape(img, (dim1, dim2, dim3))
	return img_reshape

def data_iterator(faces, labels, batch_size):
	
	batch_idx = 0
	while True:
		length = len(faces)
		idx = np.arange(0, length)
		np.random.shuffle(idx)
		
		for idx in range(0, length, batch_size):
			cur_idx = idx[idx:idx+batch_size]
			face_batch=face[cur_idx]
			label_batch = labels[cur_idx]
			yield face_batch, label_batch


def sigmoid(x, shift, mult):
	return 1 / (1 + np.exp(-(x + shift)*mult))


def norm(x):
	return (x - np.min(x)) / np.max(x - np.min(x))
	
