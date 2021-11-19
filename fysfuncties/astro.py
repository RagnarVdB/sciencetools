from astropy.io import fits
import numpy as np
import pandas as pd
import os

def open_images(folder):
    data = []
    for filename in os.listdir(folder):
        f = folder + "/" + filename
        if f[-4:] == "fits":
            with fits.open(f) as hdu:
                el = dict(hdu[0].header)
                el["filename"] = filename
                el["data"] = hdu[0].data
                data.append(el)
    return pd.DataFrame(data)