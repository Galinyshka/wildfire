import os
import warnings
import rasterio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from meteostat import Point, Daily
from tqdm import tqdm

def getBox(path_tiff):
    with rasterio.open(path_tiff) as src:
        bbox = src.bounds
        long = bbox.left
        lat = bbox.top
        bottom = -bbox.bottom
        right = bbox.right

        coeff = 1 / (2 * np.pi / 360 * 6378.137) / 1000

        new_lat = lat + bottom * coeff
        new_long = long + right * coeff / np.cos(lat * (np.pi / 180))

        res = [lat, new_lat, long, float(new_long)]

    return res

def getBands(path_tiff, key_id):
    res = pd.DataFrame(columns=['date', 'sector_id', 'lat', 'lon', 'B02', 'B03', 'B04', 'fire'])
    box = getBox(path_tiff)

    with rasterio.open(path_tiff) as src:
        blue = src.read(1)
        green = src.read(2)
        red = src.read(3)
        mask = src.read(5)

    rows = blue.shape[0]
    cols = blue.shape[1]

    pixel_lat_coef = (box[1] - box[0]) / rows
    pixel_lon_coef = (box[3] - box[2]) / cols
    
    res['B02'] = red.flatten()
    res['B03'] = green.flatten()
    res['B04'] = blue.flatten()
    res['fire'] = mask.flatten()
    res['date'] = path_tiff.split('/')[-1].split('.')[0]
    res['sector_id'] = key_id
    
    res['lat'] = np.repeat(np.linspace(box[0], box[1], num=rows), cols)
    res['lon'] = np.array([np.linspace(box[2], box[3], num=cols) for _ in range(rows)]).flatten()
    
    return res

def getMergedDataset():
    df = pd.DataFrame(columns=['date', 'sector_id', 'lat', 'lon', 'B02', 'B03', 'B04', 'fire'])

    for i in range(21):
        path_to_tiff = f'./Train data/{i}/' + list(filter(lambda x : '.tiff' in x,os.listdir(f'./Train data/{i}/')))[0]
        df = pd.concat([df, getBands(path_to_tiff, i)], axis=0)

    return df.reset_index(drop=True)