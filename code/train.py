import collections
from typing import Any
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from functools import partial
import wandb
from data import get_datasets, H, W, NUM_CHANNELS
from models import CombinedModel, ResNet1d18, ResNet18

NUM_CLASSES = 2

class TrainState(train_state.TrainState):
    batch_stats: Any

def get_metrics(logits: jnp.ndarray, labels: jnp.ndarray):
    labels = jnp.argmax(labels, axis=-1)
    logits = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(labels == logits)
    mad = jnp.mean(jnp.abs(labels - logits))
    return accuracy, mad

def get_accuracy_kth(logits: jnp.ndarray, labels: jnp.ndarray, first_k: int):
    idx_sorted_labels = jnp.argsort(labels, axis=-1, descending=True)
    idx_sorted_logits = jnp.argsort(logits, axis=-1, descending=True)
    labels_k = idx_sorted_labels[:, :first_k]
    logits_k = idx_sorted_logits[:, :first_k]
    accuracy_kth = jnp.mean(labels_k == logits_k)
    return accuracy_kth

def get_prediction(state, inputs, is_training=True):
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits, new_model_state = state.apply_fn(
        variables, inputs,
        is_training=is_training, mutable=['batch_stats'],
    )
    return new_model_state, logits

@partial(jax.jit, static_argnames=['is_training'])
def apply_model(state, inputs, labels, is_training=True):
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
    accuracy, mad = get_metrics(logits, labels)
    accuracy_kth = get_accuracy_kth(logits, labels, first_k=1)
    if is_training is False:
        return (
            grads,
            new_model_state['batch_stats'],
            loss,
            accuracy,
            mad,
            accuracy_kth,
            losses,
        )
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
    train_ds_size = len(train_ds['timeseries'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['timeseries']))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_metrics = collections.defaultdict(list)

    for perm in perms:
        batch_timeseries = train_ds['timeseries'][perm, ...]
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        batch_inputs = (batch_timeseries, batch_images)
        grads, batch_stats, loss, accuracy, mad, accuracy_kth = apply_model(
            state, batch_inputs, batch_labels
        )
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

def create_train_state(rng, use_images: bool, train_timeseries_shape):
    model = CombinedModel(
        timeseries_model_cls=partial(ResNet1d18, num_filters=8),
        images_model_cls=partial(ResNet18, num_filters=8),
        num_classes=NUM_CLASSES,
        use_images=use_images,
    )
    dummy_inputs = (
        jnp.ones([1, train_timeseries_shape]),
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

def last_eval(best_state, valid_ds):
    # Evaluate the best state on the validation set
    valid_inputs = (valid_ds['timeseries'], valid_ds['image'])
    _, _, best_valid_loss, best_valid_acc, best_valid_mad, best_valid_acc_kth, _ = apply_model(
        best_state, valid_inputs, valid_ds['label'], is_training=False
    )

    # Print the metrics of the best state
    print(
        '\nBest State Metrics:\n'
        'validation loss: %.4f valid_acc: %2f valid_mad: %2f valid_kth_acc: %2f'
        % (
            best_valid_loss,
            best_valid_acc,
            best_valid_mad,
            best_valid_acc_kth,
        )
    )

def train_and_evaluate(num_epochs: int, batch_size: int, use_images: bool, train_ds=None, valid_ds=None, seq_length=None):
    # Initialize random number generators
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    
    # Initialize the model state
    state = create_train_state(init_rng, use_images, seq_length)
    
    # Variable to track the best state based on validation loss
    best_state = state
    best_valid_loss = float('inf')  # Set initial best validation loss to a large value
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        rng, input_rng = jax.random.split(rng)
        
        # Train for one epoch
        state, train_loss, train_acc, train_mad, train_acc_kth = train_epoch(
            state, train_ds, batch_size, input_rng
        )
        
        # Prepare validation inputs and apply model
        valid_inputs = (valid_ds['timeseries'], valid_ds['image'])
        _, _, valid_loss, valid_acc, valid_mad, valid_acc_kth, losses = apply_model(
            state, valid_inputs, valid_ds['label'], is_training=False
        )
        
        # Track the best state based on validation loss
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_state = state
        
        # Print and log results every 10 epochs or at the final epoch
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
            wandb.log({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_mad": train_mad,
                "train_accuracy_kth": train_acc_kth,
                "valid_loss": valid_loss,
                "valid_accuracy": valid_acc,
                "valid_mad": valid_mad,
                "valid_accuracy_kth": valid_acc_kth,
            })

    last_eval(best_state, valid_ds)
    return best_state, losses


if __name__ == "__main__":
    wandb.init(project="floods")
    final_state, losses = train_and_evaluate(
        num_epochs=200,
        batch_size=16,
        use_images=True,
    )