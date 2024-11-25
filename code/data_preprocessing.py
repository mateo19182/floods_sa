import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class FloodDataPreprocessor:
    def __init__(self, data_dir='./data', sequence_length=730):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and prepare the training and testing data."""
        # Load CSV files
        self.train_df = pd.read_csv(os.path.join(self.data_dir, 'Train.csv'))
        self.test_df = pd.read_csv(os.path.join(self.data_dir, 'Test.csv'))
        
        # Load composite images
        self.composite_images = np.load(os.path.join(self.data_dir, 'composite_images.npz'))
        
        # Process event IDs to extract base IDs
        self.train_df['base_event_id'] = self.train_df['event_id'].apply(lambda x: x.split('_X_')[0])
        if 'label' not in self.test_df.columns:
            self.test_df['label'] = -1  # Placeholder for test data
        self.test_df['base_event_id'] = self.test_df['event_id'].apply(lambda x: x.split('_X_')[0])
        
        # Scale precipitation data
        self.scaler.fit(self.train_df[['precipitation']])
        self.train_df['precipitation_scaled'] = self.scaler.transform(self.train_df[['precipitation']])
        self.test_df['precipitation_scaled'] = self.scaler.transform(self.test_df[['precipitation']])
        
    def prepare_sequence_data(self, df, is_test=False):
        """Prepare sequential data by grouping by base event ID."""
        sequences = []
        labels = []
        event_ids = []
        
        for base_id in df['base_event_id'].unique():
            event_data = df[df['base_event_id'] == base_id].sort_values('event_id')
            
            # Get precipitation sequence
            precip_seq = event_data['precipitation_scaled'].values
            
            # Get image data if available
            image_data = None
            if base_id in self.composite_images.files:
                image_data = self.composite_images[base_id]
            
            # Get label (maximum label for the sequence)
            label = event_data['label'].max() if not is_test else -1
            
            sequences.append({
                'precipitation': precip_seq,
                'image': image_data,
                'event_id': base_id
            })
            labels.append(label)
            event_ids.append(base_id)
        
        return sequences, labels, event_ids

class FloodDataset(Dataset):
    def __init__(self, sequences, labels, event_ids, sequence_length=730):
        self.sequences = sequences
        self.labels = labels
        self.event_ids = event_ids
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        
        # Prepare precipitation data
        precip_data = torch.FloatTensor(sequence['precipitation'])
        
        # Prepare image data if available
        image_data = torch.zeros((6, 128, 128))  # Default empty tensor with correct shape
        if sequence['image'] is not None:
            # Transpose image to (channels, height, width)
            image_data = torch.FloatTensor(sequence['image']).permute(2, 0, 1)
        
        # Prepare label
        label = torch.FloatTensor([self.labels[idx]])
        
        # Add shape check for debugging
        assert image_data.shape == (6, 128, 128), f"Incorrect image shape: {image_data.shape}"
        assert precip_data.shape[0] == self.sequence_length, \
            f"Incorrect precipitation sequence length: {precip_data.shape[0]}"
        
        return {
            'precipitation': precip_data,
            'image': image_data,
            'label': label,
            'event_id': self.event_ids[idx]
        }



def create_data_loaders(preprocessor, batch_size=32, val_size=0.2):
    """Create train and validation data loaders."""
    # Load and preprocess data
    preprocessor.load_data()
    
    # Prepare sequences
    train_sequences, train_labels, train_event_ids = preprocessor.prepare_sequence_data(
        preprocessor.train_df, is_test=False
    )
    
    # Split into train and validation
    train_seq, val_seq, train_lab, val_lab, train_ids, val_ids = train_test_split(
        train_sequences, train_labels, train_event_ids,
        test_size=val_size, random_state=42, stratify=train_labels
    )
    
    # Create datasets
    train_dataset = FloodDataset(train_seq, train_lab, train_ids)
    val_dataset = FloodDataset(val_seq, val_lab, val_ids)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = FloodDataPreprocessor(data_dir='./data')
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(preprocessor, batch_size=32)
    
    # Print sample batch
    for batch in train_loader:
        print("Precipitation shape:", batch['precipitation'].shape)
        print("Image shape:", batch['image'].shape)
        print("Label shape:", batch['label'].shape)
        print("Sample event IDs:", batch['event_id'][:3])
        break