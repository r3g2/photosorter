import os
import cv2
import numpy as np
from utils import *
import matplotlib.pyplot as plt
from features import *


class Photo:

    def __init__(self,fname,data,cur_dir,color=True):
        self.fname = fname
        self.data = data
        self.color = color
        self.cur_dir = cur_dir

    def set_features(self,feature_vec):
        self.features = feature_vec



class PhotoSorter:

    def __init__(self,image_dir):
        self.image_dir = image_dir
        self.photos = []

    def _get_images(self, full_sized=False):
        if full_sized:
            image_fnames = os.listdir(self.image_dir)
            return [cv2.imread(os.path.join(self.image_dir,fname)) for fname in image_fnames]
        else:
            return [p.data for p in self.photos]

    # def _set_photo_features(self,features):
    #     rows,cols = features.shape
    #     if rows != len(self.photos):
    #         print("Can't do this operation because the")
    #     for i in range(len(photos))

    def load_images(self,size=(224,224),resize_method=cv2.INTER_LINEAR):
        image_fnames = os.listdir(self.image_dir)
        photos = []
        for fname in image_fnames:
            image_data = cv2.imread(os.path.join(self.image_dir,fname))
            resized_data = cv2.resize(image_data,size,interpolation=resize_method)
            color = True if resized_data.shape[2] == 3 else False
            photos.append(Photo(fname,resized_data,self.image_dir,color=color))
        self.photos = photos
        return photos
    
    def resize_diff(self):
        images = [p.data for p in self.photos]
        diff_matrix = np.zeros((len(images),len(images)))
        np.fill_diagonal(diff_matrix,-1)
        for i in range(len(images)-1):
            for j in range(i+1,len(images)):
                if images[i].shape == images[j].shape:
                    diff = np.abs(images[i] - images[j]).sum()
                    diff_matrix[i,j] = diff
                    diff_matrix[j,i] = diff
                else:
                    diff_matrix[i,j] = np.float('inf')
                    diff_matrix[j,i] = np.float('inf')
        return diff_matrix

    def get_duplicates(self,diff_matrix,hist_threshold=3):
        indices = np.triu_indices_from(diff_matrix,k=1)
        vals = diff_matrix[indices]
        hist = np.histogram(vals)
        small_diff_indices = diff_matrix < hist[1][hist_threshold]
        dups = []
        for i in range(len(small_diff_indices)-1):
            for j in range(i+1,len(small_diff_indices)):
                if small_diff_indices[i,j]:
                    dups.append((i,j))
        return dups



    def display_duplicates(self, duplicates,change_color=True):
        images = self._get_images(full_sized = True)
        if change_color:
            images = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in images]
        rows = len(duplicates)
        cols = 2
        dup_images = get_duplicate_images(images,duplicates)
        dup_images = [item for t in dup_images for item in t]
        fig=plt.figure(figsize=(12, 12))
        for i in range(1,cols*rows + 1):
            fig.add_subplot(rows,cols,i)
            plt.imshow(dup_images[i-1])
        plt.show()
        
    
    def compare_structural_similarity(self):
        images = self._get_images()
        sim_list = []
        for i in range(len(images)):
            s_i = []
            for j in range(i+1,len(images)):
                img1 = images[i]
                img2 = images[j]
                img1_shape = img1.shape
                img2_shape = img2.shape
                if img1_shape == img2_shape:
                    color = True if img1_shape[2] > 1 else False
                    sscore,_ = similarity(images[i],images[j],color)
                    s_i.append((j,sscore))
            sim_list.append(s_i)
        return sim_list
    
    def find_duplicate_images(self,std_devs=2.5):
        sim_list = self.compare_structural_similarity()
        sscores = get_sscores_from_sims(sim_list)
        threshold = find_threshold(sscores,std_devs=std_devs)
        dups = find_similar_images(sim_list,threshold)
        combined_dups = combine_dups(dups)
        dup_fnames = []
        for l in combined_dups:
            dup_l = []
            for ind in l:
                dup_l.append(self.photos[ind].fname)
            dup_fnames.append(dup_l)
        return dup_fnames,combined_dups
    
    def query_similar(self,query_image,num_sims = 10):
        sscores = []
        iscolor = is_color(query_image)
        images = self._get_images()
        size = images[0].shape[:2]
        if size != query_image.shape[:2]:
            query_image = cv2.resize(query_image,size)
        for i,img in enumerate(images):
            if iscolor == is_color(img):
                sscore, _ = similarity(query_image,img,color=iscolor)
                sscores.append((self.photos[i],sscore))
        top_sims = sorted(sscores,reverse=True,key=lambda x: x[1])[:num_sims]
        return top_sims

        




        

#################### TESTING ##########################
def test_query_similar():
    data_path = "/Users/rickgentry/Spring 2021/computer_vision/final_project/data/test_data"
    ps = PhotoSorter(data_path)
    ps.load_images()
    top_sims = ps.query_similar(ps.photos[1].data)
    show_images(2,5,[sim[0].data for sim in top_sims])


def test_get_image_clusters():
    data_path = "/Users/rickgentry/Spring 2021/computer_vision/final_project/data/color_test_data"
    kmeans,features = cluster_images(data_path)
    groups = get_image_clusters(data_path,kmeans)
    images = [cv2.imread(fname) for fname in groups[0]]
    show_images(1,len(images), images)


def test_find_duplicate_images():
    data_path = "/Users/rickgentry/Spring 2021/computer_vision/final_project/data/test_data"
    ps = PhotoSorter(data_path)
    ps.load_images()
    dup_fnames,dup_indices = ps.find_duplicate_images()
    photos = [ps.photos[ind].data for ind in dup_indices[1]]
    show_images(1,len(dup_fnames[1]),photos)


def test_compare_structural_similarity():
    data_path = "/Users/rickgentry/Spring 2021/computer_vision/final_project/data/test_data"
    ps = PhotoSorter(data_path)
    ps.load_images()
    sim_list = ps.compare_structural_similarity()
    sscores = get_sscores_from_sims(sim_list)
    threshold = find_threshold(sscores,std_devs=2)
    dups = find_similar_images(sim_list,threshold)
    combined_dups = combine_dups(dups)


def test_resize_duplicates():
    data_path = "/Users/rickgentry/Spring 2021/computer_vision/final_project/data/test_data"
    ps = PhotoSorter(data_path)
    ps.load_images()
    diff_matrix = ps.resize_diff()
    duplicates = ps.get_duplicates(diff_matrix)
    ps.display_duplicates(duplicates[:5])

if __name__ == "__main__":
    test_query_similar()



