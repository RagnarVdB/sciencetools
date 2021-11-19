from astropy.io import fits
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import math
from scipy import optimize as opt

def open_images(folder):
    data = []
    for filename in os.listdir(folder):
        f = folder + "/" + filename
        if f[-4:] == "fits":
            with fits.open(f) as hdu:
                el = dict(hdu[0].header)
                el["filename"] = filename
                el["data"] = hdu[0].data
                el["shape"] = el["data"].shape
                data.append(el)
    images = pd.DataFrame(data)
    try:
        images.DATE = pd.to_datetime(images.DATE)
        images.sort_values(by="DATE", inplace=True)
        images.reset_index(inplace=True)
    except:
        print("date not found")
    return images


def plot_images(images, column, dr, aspect=3, dpi=100, title=None):
    if type(images) == np.ndarray and len(images.shape) == 2:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        med = np.median(images)
        ax.imshow(images, vmin=med-dr, vmax=med+dr, origin="lower")
    elif type(images) == pd.DataFrame:
        l = math.ceil(len(images)/3)
        fig, axes = plt.subplots(l, 3, figsize=(15, 15/aspect * math.ceil(l/3)), dpi=dpi)
        axs = axes.flatten()
        meds = []
        j=0
        for i, im in images.iterrows():
            med = np.median(im[column])
            meds.append(med)
            axs[j].imshow(im[column], vmin=med-dr, vmax=med+dr, origin="lower")
            if title is not None:
                axs[j].set_title(im[title])
            j+=1
        
        while j < len(axs):
            fig.delaxes(axs[j])
            j+=1

        fig.tight_layout()
        #return (meds)


def zoom(image, center, size):
    x, y = center
    dx = int(size[0]/2)
    dy = int(size[1]/2)
    return image[x-dx:x+dx, y-dy:y+dy]


def find_crop(image):

    def score(mask, x1, x2, y1, y2):
        o = np.indices(mask.shape)
        frame = np.array(np.logical_and(np.logical_and(o[0,:,:]>x1, o[0,:,:]<x2), np.logical_and(o[1,:,:]>y1, o[1,:,:]<y2))) # IN

        return - np.logical_and(mask, frame).sum() - np.logical_and(np.logical_not(mask), np.logical_not(frame)).sum()


    s = image.shape
    mask = image > np.mean(image)
    f = lambda x: score(mask, *x)
    params = opt.minimize(f, (50, s[0]-50, 50, s[1]-50), bounds=((0, s[0]/2), (s[0]/2, s[0]), (0, s[1]/2), (s[1]/2, s[1])), method="Powell")
    return np.array(params["x"], dtype=np.uint16)

def crop(image):
    x1, x2, y1, y2 = find_crop(image)
    return image[x1:x2,y1:y2]


def make_master(frames, column):
    return np.median(np.array(list(frames[column])), axis=0)

def reduce(images, old_column, new_column, master_bias, master_flat, master_dark):
    if type(images) == pd.DataFrame:
        images[new_column] = (images[old_column] - master_bias)