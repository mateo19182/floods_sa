import torch
import torch.nn as nn

class SpatialEncoder(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.output_dim = hidden_dim * 4

    def forward(self, x):
        # Add shape check
        batch_size, channels, height, width = x.shape
        assert channels == 6, f"Expected 6 channels, got {channels}"
        assert height == 128 and width == 128, f"Expected 128x128 image, got {height}x{width}"
        
        x = self.cnn(x)
        return x.squeeze(-1).squeeze(-1)

class TemporalEncoder(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        self.output_dim = hidden_dim * 2

    def forward(self, x):
        # Add shape check
        batch_size, seq_len = x.shape
        x = x.unsqueeze(-1)  # Add feature dimension
        assert seq_len == 730, f"Expected sequence length 730, got {seq_len}"
        
        outputs, (hidden, _) = self.lstm(x)
        return torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)

class FloodDetectionModel(nn.Module):
    def __init__(self, spatial_dim=64, temporal_dim=64):
        super().__init__()
        self.spatial_encoder = SpatialEncoder(hidden_dim=spatial_dim)
        self.temporal_encoder = TemporalEncoder(hidden_dim=temporal_dim)
        
        combined_dim = self.spatial_encoder.output_dim + self.temporal_encoder.output_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, precipitation, image):
        # Add shape checks
        assert len(precipitation.shape) == 2, \
            f"Expected precipitation shape (batch_size, seq_len), got {precipitation.shape}"
        assert len(image.shape) == 4, \
            f"Expected image shape (batch_size, channels, height, width), got {image.shape}"
            
        # Encode temporal data
        temp_features = self.temporal_encoder(precipitation)
        
        # Encode spatial data
        spatial_features = self.spatial_encoder(image)
        
        # Combine features
        combined = torch.cat((temp_features, spatial_features), dim=1)
        
        # Classify
        return self.classifier(combined)