import torch
from models import FloodDetectionModel
from train import Trainer
from data_preprocessing import FloodDataPreprocessor, create_data_loaders
import wandb


def print_batch_info(batch):
    print("\nBatch information:")
    print(f"Precipitation shape: {batch['precipitation'].shape}")
    print(f"Image shape: {batch['image'].shape}")
    print(f"Labels shape: {batch['label'].shape}")
    print(f"Number of positive samples: {batch['label'].sum().item()}")
    print(f"Precipitation range: [{batch['precipitation'].min():.3f}, {batch['precipitation'].max():.3f}]")
    print(f"Image range: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")


def main():
    # Initialize wandb
    config = {
        "learning_rate": 3e-5*100,  # Reduced learning rate for stability
        "epochs": 100,          # Increased epochs since we have small number of sequences
        "batch_size": 16,       # Smaller batch size for better generalization
        "spatial_dim": 64,
        "temporal_dim": 64,
    }
    
    wandb.init(
        project="flood-detection",
        config=config
    )
    
    # Data preparation with class balancing
    preprocessor = FloodDataPreprocessor(data_dir='./data')
    train_loader, val_loader = create_data_loaders(
        preprocessor,
        batch_size=config['batch_size'],
        val_size=0.2
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Model initialization
    model = FloodDetectionModel(
        spatial_dim=config['spatial_dim'],
        temporal_dim=config['temporal_dim']
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
    )
    
    # Train model
    trainer.train(num_epochs=config['epochs'])

if __name__ == "__main__":
    main()