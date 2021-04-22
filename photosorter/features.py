##################### IMPORTS #######################
import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
#from PhotoSorter import *
from pathlib import Path
from utils import show_images
from skimage.metrics import structural_similarity
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize as norm_features

####################################################

def get_image_from_file(filename):
    '''
    Gets an image from a file for tensorflow dataset
    '''
    file = tf.io.read_file(filename)
    image = tf.image.decode_image(file)
    return {'image': image, 'fname': filename}


def create_dataset(data_dir,size=[224,224]):
    '''
    Creates a dataset for tensorflow model
    '''
    files = [os.path.join(data_dir,fname) for fname in os.listdir(data_dir)]
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(get_image_from_file)
    return dataset

def build_feature_extractor(img_shape=(224,224,3)):
    '''
    Builds a pre-trained CNN feature extraction model
    '''
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,include_top=False,weights='imagenet')
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    inputs = tf.keras.Input(shape=img_shape)
    x = preprocess_input(inputs)
    x = base_model(x,training=False)
    outputs = global_average_layer(x)
    model = tf.keras.Model(inputs,outputs)
    return model

def extract_features(model,ds):
    '''
    Takes in a feature extraction network and predicts on the dataset to return features
    '''
    img_features = []
    for x in iter(ds):
        img = tf.expand_dims(tf.image.resize(x['image'],[224,224]),axis=0)
        if img.shape.as_list() == [1,224,224,3]:
    
            features = model.predict(img,batch_size=1)
            img_features.append((x['fname'].numpy().decode('utf-8'),features))
    return img_features

def cluster(features,n_clusters=10,normalize=True):
    '''
    Performs kmeans clustering on features of the images to group them
    '''
    if normalize:
        features = norm_features(features,axis=0)
    km = KMeans(n_clusters=n_clusters).fit(features)
    return km

def apply_PCA(features,n_components=100):
    pca = PCA(n_components=n_components)
    pca.fit(features)
    pca_features = pca.transform(features)
    return pca_features

def cluster_images(dataset_dir,img_size=(224,224)):
    img_shape = img_size + (3,)
    model = build_feature_extractor(img_shape)
    dataset = create_dataset(dataset_dir,size=list(img_size))
    features = extract_features(model,dataset)
    feature_matrix = np.stack([t[1].squeeze() for t in features],axis=0)
    kmeans = cluster(feature_matrix)
    return kmeans, features

def get_image_clusters(file_paths,kmeans):
    num_groups = kmeans.n_clusters
    group_l = [[] for i in range(num_groups)]
    for i in range(len(kmeans.labels_)):
        group = kmeans.labels_[i]
        group_l[group].append(os.path.basename(file_paths[i]))
    return group_l


def extract_orb_features(image,nfeatures=500,size=(256,256)):
    '''
    Extracting keypoints and descriptors using ORB
    '''
    if image.shape[2] > 1:
        cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #image = cv2.resize(image,size)
    orb=cv2.ORB_create(nfeatures = nfeatures, scoreType=cv2.ORB_FAST_SCORE)
    keypoints,descriptors = orb.detectAndCompute(image,None)
    
    return keypoints, descriptors


def similarity(img1,img2, color=True, full=False):
    '''
    Structural similarity computation
    '''
    if full:
        mean_sim,diff_mat = structural_similarity(img1,img2,full=full, multichannel=color,gaussian_weights=True,sigma=1.5)
    else: 
        diff_mat = None
        mean_sim = structural_similarity(img1,img2,full=full, multichannel=color,gaussian_weights=True,sigma=1.5)
    return mean_sim,diff_mat



def feature_comparison(des1,des2):
    '''
    Finds good matches between descriptors
    '''
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(des1,des2,k=2)
    good_matches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good_matches.append([m])
    return good_matches



 #################### TESTING ##############################
def test_extract_orb_features(size = (128,128)):
    data_dir = ("/Users/rickgentry/Spring 2021/computer_vision/final_project/data/PhotoSorter_images")
    img1 = cv2.imread(os.path.join(data_dir,"20160601_111919.jpg"))
    img2 = cv2.imread(os.path.join(data_dir,"20160601_111922.jpg"))
    img1_features = extract_orb_features(img1)
    img2_features = extract_orb_features(img2)
    good_matches = feature_comparison(img1_features[1],img2_features[1])
    print(len(img1_features[0]))
    print(len(good_matches))
    img3 = cv2.drawMatchesKnn(img1,img1_features[0],img2,img2_features[0],good_matches,None,flags=2)
    plt.imshow(img3)
    plt.show()

def main():
    data_path = "/Users/rickgentry/Spring 2021/computer_vision/final_project/data/PhotoSorter_images"
    ps = PhotoSorter(data_path)
    ps.load_images()
    if ps.photos:
        features = []
        for photo in ps.photos:

            features.append(extract_orb_features(photo.data))
    num_good_matches = 0
    matches = []
    for i in range(len(ps.photos)):
        good_matches = feature_comparison(features[0][1],features[i][1])
        num_good_matches = len(good_matches)

        matches.append(num_good_matches) 
        top_indices = sorted(range(len(matches)), key=lambda i: matches[i], reverse=True)[:10]

    closest_imgs = [ps.photos[ind].data for ind in top_indices]
    show_images(2,5,closest_imgs)



if __name__ == "__main__":
    main()