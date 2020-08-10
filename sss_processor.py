import scipy.io
from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigs
import random
from PIL import Image
# %matplotlib inline
from scipy.sparse import csr_matrix
import os



def load_image_feat_array(image_path,feature_path):
	image = np.asarray(plt.imread(image_path))
	feat = np.asarray(scipy.io.loadmat(feature_path)['embedmap'])
	print(f'shape of image array is',image.shape)
	print(f'shape of feature array is',feat.shape)
	return image, feat


def pca_feature(image,feature):
	# Filter out super high numbers due to some instability in the network
	feature[feature>5] = 5
	feature[feature<-5] = -5
	#### Missing an image guided filter with the image as input
	##
	##########
	# change to double precision
	feature = np.float64(feature)
	# retrieve size of feature array
	shape = feature.shape
	[h, w, d] = feature.shape
	# resize to a two-dimensional array
	feature = np.reshape(feature, (h*w,d))
	# calculate average of each column
	featmean = np.average(feature,0)
	onearray = np.ones((h*w,1))
	featmeanarray = np.multiply(np.ones((h*w,1)),featmean)
	feature = np.subtract(feature,featmeanarray)
	feature_transpose = np.transpose(feature)
	cover = np.dot(feature_transpose, feature)
	# get largest eigenvectors of the array
	val,vecs = eigs(cover, k=3, which='LI')
	pcafeature = np.dot(feature, vecs)
	pcafeature = np.reshape(pcafeature,(h,w,3))
	pcafeature = np.float64(pcafeature)
	return pcafeature


def normalise_0_1(feature):
	max_value = np.amin(feature)
	min_value = np.amax(feature)
	subtract = max_value - min_value
	for i in range(0,3):
		feature[:,:,i] = feature[:,:,i] - np.amin(feature[:,:,i])
		feature[:,:,i] = feature[:,:,i] / np.amax(feature[:,:,i])
	return feature

def extract_pure_name(original_name):
	pure_name, extention = os.path.splitext(original_name)
	return pure_name


def listdir_nohidden(path):
	new_list = []
	for f in os.listdir(path):
		if not f.startswith('.'):
			new_list.append(f)
	new_list.sort()
	return new_list



def create_folder_if_not_exist(folder_path):
	if not os.path.exists(folder_path):
		os.makedirs(folder_path)

def process_single(img_path, features_path, output_path):
	image, feature = load_image_feat_array(img_path, features_path)
	image = plt.imread(img_path)
	pca = pca_feature(image,feature)
	normalise_feature = normalise_0_1(pca)
	normalise_feature = (normalise_feature*255.0).astype(np.uint8)
	im = Image.fromarray(normalise_feature)
	im.save(output_path)

def process_folder(img_folder, features_folder, output_folder):
	img_list = listdir_nohidden(img_folder)
	index=0
	for img in img_list:
		index += 1
		print('---------')
		print(f'Index: {index}')
		print(f'Process: {img}')
		img = extract_pure_name(img)
		img_path = img_folder+img+'.png'
		features_path = features_folder+img+'.mat'
		output_path = output_folder+img+'.jpg'
		process_single(img_path, features_path, output_path)


# if __name__ == '__main__':
img_folder = '/original_img/'
features_folder ='/feature_mat/'
output_folder = '/feature_jpg/'
create_folder_if_not_exist(output_folder)
process_folder(img_folder, features_folder, output_folder)
