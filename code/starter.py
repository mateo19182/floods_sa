
# @title Imports

import collections
from collections.abc import Callable, Sequence
from concurrent import futures
from datetime import datetime
from functools import partial
import math
import multiprocessing
import os
from typing import Any

import flax
from flax import linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import optax
import pandas as pd
import seaborn as sns
import skimage as ski
from tqdm import tqdm

# from google.colab import drive
# from google.colab import widgets


# Replace this path with your path
BASE_PATH = 'data/'  #@param {type: 'string'}

data = pd.read_csv(os.path.join(BASE_PATH, 'Train.csv'))
data_test = pd.read_csv(os.path.join(BASE_PATH, 'Test.csv'))

data['event_id'] = data['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
data['event_idx'] = data.groupby('event_id', sort=False).ngroup()
data_test['event_id'] = data_test['event_id'].apply(lambda x: '_'.join(x.split('_')[0:2]))
data_test['event_idx'] = data_test.groupby('event_id', sort=False).ngroup()

data['event_t'] = data.groupby('event_id').cumcount()
data_test['event_t'] = data_test.groupby('event_id').cumcount()

print(data.head())
print(data_test.head())

len(data[data.event_id == 'id_p8f40663jj3g'])

images_path = os.path.join(BASE_PATH, 'composite_images.npz')
images = np.load(images_path)
print(images)
print('The folder contains', len(images), 'images, both for train and test.')
print('There are', len(data['event_id'].unique()), 'train event ids and', len(data_test['event_id'].unique()), 'test event ids.')

# Image metadata constants

# 5 bands and the slope of each image
BAND_NAMES =  ('B2', 'B3', 'B4', 'B8', 'B11', 'slope')
# Image shape
H, W, NUM_CHANNELS = IMG_DIM = (128, 128, len(BAND_NAMES))

event_id = 'id_rhg5w8vmv3ny'

num_cols = len(BAND_NAMES)
_, axes = plt.subplots(
    ncols=num_cols,
    figsize=(num_cols * 3.5, 3.5),
    facecolor='white',
)

for band_idx in range(num_cols):
  img = images[event_id][..., band_idx]
  axes[band_idx].imshow(img, cmap='gray', interpolation='nearest')
  axes[band_idx].set_title(f'band {BAND_NAMES[band_idx]}')
  axes[band_idx].get_xaxis().set_visible(False)
  axes[band_idx].get_yaxis().set_visible(False)

sample_image = next(iter(images.values()))
assert sample_image.shape == IMG_DIM
assert sample_image.dtype == np.uint16
_MAX_INT = np.iinfo(np.uint16).max

def decode_slope(x: np.ndarray) -> np.ndarray:
  # Convert 16-bit discretized slope to float32 radians
  return (x / _MAX_INT * (math.pi / 2.0)).astype(np.float32)

def normalize(x: np.ndarray, mean: int, std: int) -> np.ndarray:
  return (x - mean) / std

rough_S2_normalize = partial(normalize, mean=1250, std=500)

def preprocess_image(x: np.ndarray) -> np.ndarray:
  return np.concatenate([
      rough_S2_normalize(x[..., :-1].astype(np.float32)),
      decode_slope(x[..., -1:]),
  ], axis=-1, dtype=np.float32)


rng = np.random.default_rng(seed=0xf100d)

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

# For the test set there are no labels
test_timeseries = data_test.pivot(index='event_id', columns='event_t', values='precipitation').to_numpy()

event_splits = data_new.groupby('event_id')['split'].first()

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

print(f'{train_timeseries.shape=}')
print(f'    {train_images.shape=}')
print(f'    {train_labels.shape=}')

print(f'{valid_timeseries.shape=}')
print(f'    {valid_images.shape=}')
print(f'    {valid_labels.shape=}')

print(f'{test_timeseries.shape=}')
print(f'    {test_images.shape=}')


"""Flax implementation of ResNet V1.5, without output head to return embedding."""
ModuleDef = Any


class ResNetBlock(nn.Module):
  """ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(
      self,
      x,
  ):
    residual = x
    y = self.conv(self.filters, (3, 3), self.strides)(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3, 3))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters, (1, 1), self.strides, name='conv_proj'
      )(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet(nn.Module):
  """ResNetV1.5."""

  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv

  @nn.compact
  def __call__(self, x, is_training: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(
        nn.BatchNorm,
        use_running_average=not is_training,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
    )

    x = conv(
        self.num_filters,
        (7, 7),
        (2, 2),
        padding=[(3, 3), (3, 3)],
        name='conv_init',
    )(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
        )(x)
    x = jnp.mean(x, axis=(1, 2))
    return x

ResNet18 = partial(ResNet, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock)

"""Adaptation of ResNet for 1D time series, without output head (returns embedding)."""

class ResNetBlock1d(nn.Module):
  """ResNet block."""

  filters: int
  conv: ModuleDef
  norm: ModuleDef
  act: Callable
  strides: tuple[int] = (1,)

  @nn.compact
  def __call__(
      self,
      x,
  ):
    residual = x
    y = self.conv(self.filters, (3,), self.strides, padding='SAME')(x)
    y = self.norm()(y)
    y = self.act(y)
    y = self.conv(self.filters, (3,), padding='SAME')(y)
    y = self.norm(scale_init=nn.initializers.zeros_init())(y)

    if residual.shape != y.shape:
      residual = self.conv(
          self.filters, (1,), self.strides, name='conv_proj'
      )(residual)
      residual = self.norm(name='norm_proj')(residual)

    return self.act(residual + y)


class ResNet1d(nn.Module):
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  embed: bool = True

  @nn.compact
  def __call__(self, x, is_training: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(
        nn.BatchNorm,
        use_running_average=not is_training,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
    )
    x = jnp.expand_dims(x, axis=-1)  # add a virtual 'channel' dimension
    x = conv(
        self.num_filters,
        (7,),
        (1,),
        padding='SAME',
        name='conv_init',
    )(x)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3,), strides=(1,), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        x = self.block_cls(
            self.num_filters * 2**i,
            strides=(1,),
            conv=conv,
            norm=norm,
            act=self.act,
        )(x)
    # linear project down to a single logit per timestep
    return x  # (B, T, C)

ResNet1d18 = partial(ResNet1d, stage_sizes=[2, 2, 2, 2], block_cls=ResNetBlock1d)

"""Combined time series + image model."""

class CombinedModel(nn.Module):
  """Takes two embedding models and passes them through a single output head."""
  timeseries_model_cls: ModuleDef
  images_model_cls: ModuleDef
  num_classes: int
  dtype: Any = jnp.float32
  use_images: bool = True

  @nn.compact
  def __call__(
      self, x: tuple[jnp.ndarray, jnp.ndarray], is_training: bool = True
  ):
    x_timeseries, x_images = x
    T = x_timeseries.shape[1]

    timeseries_model = self.timeseries_model_cls()
    logits = timeseries_model(x_timeseries, is_training=is_training)  # (B, T, C)

    if self.use_images:
      images_model = self.images_model_cls()
      x_images = images_model(x_images, is_training=is_training)
      # tile to match time axis
      x_images = jnp.tile(jnp.expand_dims(x_images, axis=1), (1, T, 1))


      logits = jnp.concatenate([logits, x_images], axis=-1)


    x = nn.Conv(1, (1,), padding='SAME', name='output_head')(logits)
    x = jnp.squeeze(x, axis=-1)
    x = jnp.asarray(x, self.dtype)
    return x

NUM_CLASSES = 2  # Flood / No flood.

def get_datasets():
  """Load train and test datasets into memory."""
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
  return train_ds, valid_ds, test_ds

def get_metrics(logits: jnp.ndarray, labels: jnp.ndarray):
  # Argmax chooses the most probable time step to be flooded
  labels = jnp.argmax(labels, axis=-1)
  logits = jnp.argmax(logits, axis=-1)
  accuracy = jnp.mean(labels == logits)
  mad = jnp.mean(jnp.abs(labels - logits))
  return accuracy, mad


def get_accuracy_kth(logits: jnp.ndarray, labels: jnp.ndarray, first_k: int):
  # This method chooses the k-th most probable time steps to be flooded.
  # If k=1 this is equivalent to accuracy from `get_metrics`
  idx_sorted_labels = jnp.argsort(labels, axis=-1, descending=True)
  idx_sorted_logits = jnp.argsort(logits, axis=-1, descending=True)
  labels_k = idx_sorted_labels[:,:first_k]
  logits_k = idx_sorted_logits[:,:first_k]
  accuracy_kth = jnp.mean(labels_k == logits_k)
  return accuracy_kth

class TrainState(train_state.TrainState):
  batch_stats: Any


def get_prediction(state, inputs, is_training=True):
  logits, new_model_state = state.apply_fn(
      {'params': state.params, 'batch_stats': state.batch_stats}, inputs,
      is_training=is_training, mutable=['batch_stats'],
      )
  return new_model_state, logits

@partial(jax.jit, static_argnames=['is_training'])
def apply_model(state, inputs, labels, is_training=True):
  """Computes gradients, loss and accuracy for a single batch."""

  def loss_fn(params):
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        inputs,
        is_training=is_training,
        mutable=['batch_stats'],
    )
    losses = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
    loss = jnp.mean(losses)
    return loss, (new_model_state, logits, losses)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, (new_model_state, logits, losses)), grads = grad_fn(state.params)
  # Logits shape is: batch size x number of time steps

  accuracy, mad = get_metrics(logits, labels)
  accuracy_kth = get_accuracy_kth(logits, labels, first_k=1)
  if is_training is False:
    # for validation
    return (
        grads,
        new_model_state['batch_stats'],
        loss,
        accuracy,
        mad,
        accuracy_kth,
        losses,)
  else:
    return (
        grads,
        new_model_state['batch_stats'],
        loss,
        accuracy,
        mad,
        accuracy_kth,
    )


@jax.jit
def update_model(state, grads, batch_stats):
  return state.apply_gradients(grads=grads, batch_stats=batch_stats)


def train_epoch(state, train_ds, batch_size, rng):
  """Train for a single epoch."""
  train_ds_size = len(train_ds['timeseries'])
  steps_per_epoch = train_ds_size // batch_size

  perms = jax.random.permutation(rng, len(train_ds['timeseries']))
  perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
  perms = perms.reshape((steps_per_epoch, batch_size))

  epoch_metrics = collections.defaultdict(list)

  for perm in perms:
    batch_timeseries = train_ds['timeseries'][perm, ...]
    # print(f'{batch_timeseries.shape=}')
    batch_images = train_ds['image'][perm, ...]
    # print(f'{batch_images.shape=}')
    batch_labels = train_ds['label'][perm, ...]
    batch_inputs = (batch_timeseries, batch_images)
    grads, batch_stats, loss, accuracy, mad, accuracy_kth = apply_model(
        state, batch_inputs, batch_labels)
    state = update_model(state, grads, batch_stats)
    epoch_metrics['loss'].append(loss)
    epoch_metrics['accuracy'].append(accuracy)
    epoch_metrics['accuracy_kth'].append(accuracy_kth)
    epoch_metrics['mad'].append(mad)

  n = len(epoch_metrics['loss'])

  return (
      state,
      sum(epoch_metrics['loss']) / n,
      sum(epoch_metrics['accuracy']) / n,
      sum(epoch_metrics['mad']) / n,
      sum(epoch_metrics['accuracy_kth']) / n,
  )


def create_train_state(rng, use_images: bool):
  """Creates initial `TrainState`."""
  model = CombinedModel(
      timeseries_model_cls=partial(ResNet1d18, num_filters=8),
      images_model_cls=partial(ResNet18, num_filters=8),
      num_classes=NUM_CLASSES,
      use_images=use_images,
  )
  dummy_inputs = (
      jnp.ones([1, train_timeseries.shape[-1]]),
      jnp.ones([1, H, W, NUM_CHANNELS]),
  )
  variables = model.init(rng, dummy_inputs)
  tx = optax.adamw(1e-3)
  return TrainState.create(
      params=variables['params'],
      batch_stats=variables['batch_stats'],
      apply_fn=model.apply,
      tx=tx,
  )

def train_and_evaluate(
    num_epochs: int,
    batch_size: int,
    use_images: bool,
) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard  summaries are written to.

  Returns:
    The train state (which includes the `.params`).
  """
  train_ds, valid_ds, _ = get_datasets()
  rng = jax.random.key(0)

  rng, init_rng = jax.random.split(rng)
  state = create_train_state(init_rng, use_images)

  for epoch in range(1, num_epochs + 1):
    rng, input_rng = jax.random.split(rng)
    state, train_loss, train_acc, train_mad, train_acc_kth = (
        train_epoch(state, train_ds, batch_size, input_rng)
    )
    valid_inputs = (valid_ds['timeseries'], valid_ds['image'])
    _, _, valid_loss, valid_acc, valid_mad, valid_acc_kth, losses = (
        apply_model(
            state, valid_inputs, valid_ds['label'], is_training=False,
        )
    )

    if epoch == 1 or epoch % 10 == 0 or epoch == num_epochs:
      print(
          'epoch:% 3d \n'
          'train loss: %.4f train_acc: %2f train_mad: %2f train_kth_acc: %2f\n'
          'validation loss: %.4f valid_acc: %2f valid_mad: %2f\n valid_kth_acc: %2f'
          % (
              epoch,
              train_loss,
              train_acc,
              train_mad,
              train_acc_kth,
              valid_loss,
              valid_acc,
              valid_mad,
              valid_acc_kth,
          )
          )
  return state, losses

final_state, losses = train_and_evaluate(
    num_epochs=150,
    batch_size=64,
    use_images=True,
)

event_id = 0
x = np.arange(len(losses[event_id,:]))
plt.title('Validation set results')
plt.plot(x,losses[event_id,:], label='Predicted losses of flood', color='blue')
plt.plot(x,valid_labels[event_id,:], label='Label for flooding event',color='red')
plt.ylabel('Losses value for flood probability')
plt.xlabel('event_t')
plt.legend()
plt.show()

# Logits for the test set
_, _, test_ds = get_datasets()
test_inputs = (test_ds['timeseries'], test_ds['image'])
_, logits = get_prediction(final_state, test_inputs,is_training=False,)

plt.plot(x,logits[event_id,:], color='green')
plt.ylabel('Logit value for flood probability')
plt.title('Test set results')
plt.xlabel('event_t')
plt.show()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

probs = sigmoid(logits)
probs.shape

sample_submission = pd.read_csv(BASE_PATH + '/SampleSubmission.csv')
sample_submission['label'] = probs.flatten()
sample_submission.head()

sample_submission.to_csv('BenchmarkSubmission.csv', index = False)

