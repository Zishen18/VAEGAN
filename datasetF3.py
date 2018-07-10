import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.misc
from glob import glob
from tqdm import tqdm
import h5py
def imread(filePath):
	i = scipy.misc.imread(filePath)
	image = i.astype(np.float)
	return image

def centerCrop(image, height=64):
	h = np.shape(image)[0]
	j = int(round((h - height)/2.0))
	return image[j:j+height, :, :]

def resizeWidth(image, width=64.0):
	h, w = np.shape(image)[:2]
	shape = [int((float(h)/w)*width), width]
	i = scipy.misc.imresize(image, shape)
	return i

def getImage(path, width=64, height=64):
	i_ori = imread(path)
	i_resize = resizeWidth(i_ori, width=width)
	img = centerCrop(i_resize, height=height)
	return img

data = glob(os.path.join("../img_align_celeba", "*.jpg"))
data = np.sort(data)
print("length of data = ", len(data))

DIM = 64

images = np.zeros((len(data), DIM*DIM*3), dtype = np.uint8)

for i in tqdm(range(1500)):
	img = getImage(data[i], DIM, DIM)
	images[i] = img.flatten()
attrFile = '../list_attr_celeba.txt'

with open(attrFile, 'r') as f:
	f.readline() # skip the first line
	headers = f.readline()
headers = headers.split()
#print(headers)
#print(len(headers))	

label_input = pd.read_fwf(attrFile, skiprows=2, widths = [10,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3], index_col=0, header=None)
print(label_input)
labels = label_input.astype(int).values
print(labels)
#print(len(labels[0]))

headers_de = []
for j in headers:
	headers_de.append(j.encode())
dataFilePath = 'datasets/faces_dataset_new.h5'

with h5py.File(dataFilePath, 'w') as f:
	print('creating images dataset...')
	dataSetFace = f.create_dataset("images", data=images, dtype = np.uint8)
	print("images finished!")
	
	print('creating headers dataset...')
	dataSetHeaders = f.create_dataset("headers", data = headers_de)
	print("headers finished!")
	
	print("creating input label dataset...")
	dataSetLabelInput =  f.create_dataset("label_input", data = label_input, dtype = np.int)
	print("input label finished!")

