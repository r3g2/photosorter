import matplotlib.pyplot as plt
import cv2
import numpy as np

def show_images(rows,cols, images, figsize=(12,12),change_color=True):
    if change_color:
        images = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB) for image in images]
    fig = plt.figure(figsize=figsize)
    for i in range(cols*rows):
        fig.add_subplot(rows,cols,i+1)
        plt.imshow(images[i])
    plt.show()

def is_color(img):
    return True if (len(img.shape) == 3 and img.shape[2] == 3) else False

def get_duplicate_images(images,duplicates):
    return [(images[d[0]],images[d[1]]) for d in duplicates]


def get_sscores_from_sims(sims):
    return [t[1] for l in sims for t in l]

def find_threshold(sscores,std_devs = 3):
    sscores = np.array(sscores)
    mean_score = sscores.mean()
    std_score = sscores.std()
    return mean_score + std_devs*std_score

def combine_dups(duplicates):
    combined = []
    skip = []
    for i in range(len(duplicates)):
        if i not in skip:
            cur_combined = duplicates[i]
            for j in range(i+1, len(duplicates)):
                if any(item in cur_combined for item in duplicates[j]):
                    skip.append(j)
                    for k in range(len(duplicates[j])):
                        if duplicates[j][k] not in cur_combined:
                            cur_combined.append(duplicates[j][k])
                        
            combined.append(cur_combined)
    return combined

def find_similar_images(sims,threshold):
    dups = []
    for i in range(len(sims)):
        sim_i = sims[i]
        li = []
        for ind,score in sim_i:
            if score > threshold:
                li.append(ind)
        if li:
            li.append(i)
            dups.append(li)

    return dups