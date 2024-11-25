import os
import numpy as np
import pandas as pd
from functools import partial
import math

BASE_PATH = 'data/'

def load_data():
    data = pd.read_csv(os.path.join(BASE_PATH, 'Train.csv'))
    data_test = pd.read_csv(os.path.join(BASE_PATH, 'Test.csv'))
    
    data['event_id'] = data['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    data['event_idx'] = data.groupby('event_id', sort=False).ngroup()
    data_test['event_id'] = data_test['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
    data_test['event_idx'] = data_test.groupby('event_id', sort=False).ngroup()
    
    data['event_t'] = data.groupby('event_id').cumcount()
    data_test['event_t'] = data_test.groupby('event_id').cumcount()
    
    return data, data_test

def load_images():
    images_path = os.path.join(BASE_PATH, 'composite_images.npz')
    images = np.load(images_path)
    return images

BAND_NAMES = ('B2', 'B3', 'B4', 'B8', 'B11', 'slope')
H, W, NUM_CHANNELS = IMG_DIM = (128, 128, len(BAND_NAMES))
_MAX_INT = np.iinfo(np.uint16).max

def decode_slope(x: np.ndarray) -> np.ndarray:
    return (x / _MAX_INT * (math.pi / 2.0)).astype(np.float32)

def normalize(x: np.ndarray, mean: int, std: int) -> np.ndarray:
    return (x - mean) / std

rough_S2_normalize = partial(normalize, mean=1250, std=500)

def preprocess_image(x: np.ndarray) -> np.ndarray:
    return np.concatenate([
        rough_S2_normalize(x[..., :-1].astype(np.float32)),
        decode_slope(x[..., -1:]),
    ], axis=-1, dtype=np.float32)

def split_data(data):
    event_ids = data['event_id'].unique()
    new_split = pd.Series(
        data=np.random.choice(['train', 'valid'], size=len(event_ids), p=[0.9, 0.1]),
        index=event_ids,
        name='split',
    )
    data_new = data.join(new_split, on='event_id')
    
    train_df = data_new[(data_new['split'] == 'train')]
    train_timeseries = train_df.pivot(index='event_id', columns='event_t', values='precipitation').to_numpy()
    train_labels = train_df.pivot(index='event_id', columns='event_t', values='label').to_numpy()
    
    valid_df = data_new[data_new['split'] == 'valid']
    valid_timeseries = valid_df.pivot(index='event_id', columns='event_t', values='precipitation').to_numpy()
    valid_labels = valid_df.pivot(index='event_id', columns='event_t', values='label').to_numpy()
    
    return train_timeseries, train_labels, valid_timeseries, valid_labels

def prepare_test_data(data_test):
    test_timeseries = data_test.pivot(index='event_id', columns='event_t', values='precipitation').to_numpy()
    return test_timeseries

def prepare_images(data, data_test, images):
    event_splits = data.groupby('event_id')['split'].first()
    
    train_images = []
    valid_images = []
    test_images = []
    
    for event_id in event_splits.index:
        img = preprocess_image(images[event_id])
        if event_splits[event_id] == 'train':
            train_images.append(img)
        else:
            valid_images.append(img)
    
    for event_id in data_test['event_id'].unique():
        img = preprocess_image(images[event_id])
        test_images.append(img)
    
    train_images = np.stack(train_images, axis=0)
    valid_images = np.stack(valid_images, axis=0)
    test_images = np.stack(test_images, axis=0)
    
    return train_images, valid_images, test_images

def get_datasets():
    data, data_test = load_data()
    images = load_images()
    
    event_ids = data['event_id'].unique()
    new_split = pd.Series(
        data=np.random.choice(['train', 'valid'], size=len(event_ids), p=[0.9, 0.1]),
        index=event_ids,
        name='split',
    )
    data_new = data.join(new_split, on='event_id')

    print("Columns in data_new:", data_new.columns)

    train_df = data_new[(data_new['split'] == 'train')]
    train_timeseries = train_df.pivot(index='event_id', columns='event_t', values='precipitation').to_numpy()
    train_labels = train_df.pivot(index='event_id', columns='event_t', values='label').to_numpy()

    valid_df = data_new[data_new['split'] == 'valid']
    valid_timeseries = valid_df.pivot(index='event_id', columns='event_t', values='precipitation').to_numpy()
    valid_labels = valid_df.pivot(index='event_id', columns='event_t', values='label').to_numpy()

    test_timeseries = data_test.pivot(index='event_id', columns='event_t', values='precipitation').to_numpy()
    train_images, valid_images, test_images = prepare_images(data_new, data_test, images)
    
    train_ds = {
        'timeseries': train_timeseries,
        'image': train_images,
        'label': train_labels,
    }
    valid_ds = {
        'timeseries': valid_timeseries,
        'image': valid_images,
        'label': valid_labels,
    }
    test_ds = {
        'timeseries': test_timeseries,
        'image': test_images,
    }
    return train_ds, valid_ds, test_ds, train_timeseries.shape[-1]