import collections
from typing import Any
import os
import flax
import shutil
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
from functools import partial
import wandb
from models import CombinedModel, ResNet1d18, ResNet18
import numpy as np
import orbax.checkpoint as ocp
from data import get_datasets, H, W, NUM_CHANNELS

NUM_CLASSES = 2
CHECKPOINT_DIR = "checkpoints"

class TrainState(train_state.TrainState):
    batch_stats: Any

def calculate_metrics(logits: jnp.ndarray, labels: jnp.ndarray):
    """Calculate comprehensive metrics including F1 score"""
    predictions = jnp.where(logits > 0.5, 1, 0)
    true_positives = jnp.sum(predictions * labels)
    false_positives = jnp.sum(predictions * (1 - labels))
    false_negatives = jnp.sum((1 - predictions) * labels)
    
    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    accuracy = jnp.mean(predictions == labels)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
    }

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
        variables = {'params': params, 'batch_stats': state.batch_stats}
        logits, new_model_state = state.apply_fn(
            variables,
            inputs,
            is_training=is_training,
            mutable=['batch_stats'],
        )
        # Basic BCE loss
        bce_loss = optax.sigmoid_binary_cross_entropy(logits=logits, labels=labels)
        # Apply higher weight to positive samples (flood events)
        sample_weights = jnp.where(labels == 1, 10.0, 1.0)  # 10x weight for positive samples
        loss = jnp.mean(bce_loss * sample_weights)
        return loss, (new_model_state, logits, bce_loss)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (new_model_state, logits, losses)), grads = grad_fn(state.params)
    metrics = calculate_metrics(logits, labels)
    
    if is_training:
        return grads, new_model_state['batch_stats'], loss, metrics
    else:
        return grads, new_model_state['batch_stats'], loss, metrics, losses

@jax.jit
def update_model(state, grads, batch_stats):
    return state.apply_gradients(grads=grads, batch_stats=batch_stats)

def train_epoch(state, train_ds, batch_size, rng):
    train_ds_size = len(train_ds['timeseries'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[: steps_per_epoch * batch_size]
    perms = perms.reshape((steps_per_epoch, batch_size))

    epoch_metrics = collections.defaultdict(list)

    for perm in perms:
        batch_timeseries = train_ds['timeseries'][perm, ...]
        batch_images = train_ds['image'][perm, ...]
        batch_labels = train_ds['label'][perm, ...]
        batch_inputs = (batch_timeseries, batch_images)
        
        grads, batch_stats, loss, metrics = apply_model(
            state, batch_inputs, batch_labels
        )
        state = update_model(state, grads, batch_stats)
        
        epoch_metrics['loss'].append(loss)
        for k, v in metrics.items():
            epoch_metrics[k].append(v)

    return state, {k: jnp.mean(jnp.array(v)) for k, v in epoch_metrics.items()}

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
    
    # Learning rate schedule
    # total_steps = 150 * (train_timeseries_shape // 64)  # epochs * steps_per_epoch
    # schedule = optax.warmup_cosine_decay_schedule(
    #     init_value=0.0,
    #     peak_value=1e-3,
    #     warmup_steps=total_steps // 20,
    #     decay_steps=total_steps,
    # )
    tx = optax.adamw(learning_rate=1e-4)
    
    return TrainState.create(
        params=variables['params'],
        batch_stats=variables['batch_stats'],
        apply_fn=model.apply,
        tx=tx,
    )

def save_model(state, path):
    """Save model state with proper serialization"""
    checkpointer = ocp.StandardCheckpointer()
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)
    
    # checkpointer.save(CHECKPOINT_DIR / 'state', state)


def load_model(path):
    """Load model state"""
    checkpointer = ocp.StandardCheckpointer()
    return checkpointer.restore(path)

class EarlyStopping:
    def __init__(self, patience=25, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def train_and_evaluate(num_epochs: int, batch_size: int, use_images: bool, train_ds=None, valid_ds=None, seq_length=None):
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Initialize model
    rng = jax.random.key(0)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, use_images, seq_length)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(patience=50)
    best_valid_f1 = 0.0
    best_valid_loss = float('inf')
    losses = []

    for epoch in range(1, num_epochs + 1):
        # Training
        rng, input_rng = jax.random.split(rng)
        state, train_metrics = train_epoch(state, train_ds, batch_size, input_rng)
        
        # Validation
        valid_inputs = (valid_ds['timeseries'], valid_ds['image'])
        _, _, valid_loss, valid_metrics, _ = apply_model(
            state, valid_inputs, valid_ds['label'], is_training=False
        )
        
        # Save best model based on different metrics
        if valid_metrics['f1_score'] > best_valid_f1:
            best_valid_f1 = valid_metrics['f1_score']
            save_model(
                state,
                os.path.abspath(os.path.join(CHECKPOINT_DIR, 'best_f1_score'))
            )
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_model(
                state,
                os.path.abspath(os.path.join(CHECKPOINT_DIR, f'best_loss'))
            )
        
        # Log metrics
        metrics_dict = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'valid_loss': valid_loss,
            # 'learning_rate': state.params['learning_rate']
        }
        for k in ['accuracy', 'precision', 'recall', 'f1_score']:
            metrics_dict.update({
                f'train_{k}': train_metrics[k],
                f'valid_{k}': valid_metrics[k],
            })
        wandb.log(metrics_dict)
        
        # Print progress
        if epoch == 1 or epoch % 5 == 0 or epoch == num_epochs:
            print(
                f'\nEpoch {epoch:3d}:'
                f'\nTrain - Loss: {train_metrics["loss"]:.4f}, '
                f'F1: {train_metrics["f1_score"]:.4f}, '
                f'Precision: {train_metrics["precision"]:.4f}, '
                f'Recall: {train_metrics["recall"]:.4f}'
                f'\nValid - Loss: {valid_loss:.4f}, '
                f'F1: {valid_metrics["f1_score"]:.4f}, '
                f'Precision: {valid_metrics["precision"]:.4f}, '
                f'Recall: {valid_metrics["recall"]:.4f}'
            )
        
        # Early stopping check
        if early_stopping(valid_loss):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
        
        losses.append(valid_loss)
    
    # Save final model
    save_model(
        state,
        os.path.abspath((os.path.join(CHECKPOINT_DIR, 'final')))
    )
    
    # Save training metadata
    metadata = {
        'best_valid_f1': best_valid_f1,
        'best_valid_loss': best_valid_loss,
        'final_epoch': epoch,
    }
    
    return state, losses

if __name__ == "__main__":
    wandb.init(project="floods", config={
        "batch_size": 64,
        "num_epochs": 150,
        "use_images": True,
        "learning_rate": 1e-3,
        "pos_weight": 10.0,
    })
    
    train_ds, valid_ds, test_ds, seq_length = get_datasets(augment=True)
    
    final_state = train_and_evaluate(
        num_epochs=150,
        batch_size=64,
        use_images=True,
        train_ds=train_ds,
        valid_ds=valid_ds,
        seq_length=seq_length,
    )