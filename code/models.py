from typing import Any, Callable, Sequence
from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn

ModuleDef = Any

class ResNetBlock(nn.Module):
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1), self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)

class ResNet(nn.Module):
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

        x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name='conv_init')(x)
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

class ResNetBlock1d(nn.Module):
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: tuple[int] = (1,)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3,), self.strides, padding='SAME')(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3,), padding='SAME')(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1,), self.strides, name='conv_proj')(residual)
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
        x = conv(self.num_filters, (7,), (1,), padding='SAME', name='conv_init')(x)
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

class CombinedModel(nn.Module):
    timeseries_model_cls: ModuleDef
    images_model_cls: ModuleDef
    num_classes: int
    dtype: Any = jnp.float32
    use_images: bool = True

    @nn.compact
    def __call__(self, x: tuple[jnp.ndarray, jnp.ndarray], is_training: bool = True):
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