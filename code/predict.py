import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
import itertools
from datetime import datetime
import joblib

class FloodPredictor:
    def __init__(self, data_dir='./data', checkpoint_dir='./checkpoints'):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
    
    @staticmethod
    def get_max_consecutive(bool_array):
        return max([sum(group) for val, group in itertools.groupby(bool_array) if val], default=0)
    
    def create_features(self, df, base_id):
        """Create features for a single event sequence."""
        precipitation = df['precipitation'].values
        
        return {
            'precip_mean': np.mean(precipitation),
            'precip_std': np.std(precipitation),
            'precip_max': np.max(precipitation),
            'precip_sum': np.sum(precipitation),
            'precip_zeros': np.sum(precipitation == 0) / len(precipitation),
            
            'precip_25': np.percentile(precipitation, 25),
            'precip_50': np.percentile(precipitation, 50),
            'precip_75': np.percentile(precipitation, 75),
            'precip_90': np.percentile(precipitation, 90),
            'precip_95': np.percentile(precipitation, 95),
            'precip_99': np.percentile(precipitation, 99),
            
            'rolling_3d_max': pd.Series(precipitation).rolling(3).max().max(),
            'rolling_7d_max': pd.Series(precipitation).rolling(7).max().max(),
            'rolling_14d_max': pd.Series(precipitation).rolling(14).max().max(),
            'rolling_30d_max': pd.Series(precipitation).rolling(30).max().max(),
            
            'rolling_3d_mean': pd.Series(precipitation).rolling(3).mean().max(),
            'rolling_7d_mean': pd.Series(precipitation).rolling(7).mean().max(),
            'rolling_14d_mean': pd.Series(precipitation).rolling(14).mean().max(),
            'rolling_30d_mean': pd.Series(precipitation).rolling(30).mean().max(),
            
            'days_above_10mm': np.sum(precipitation > 10),
            'days_above_20mm': np.sum(precipitation > 20),
            'days_above_50mm': np.sum(precipitation > 50),
            'days_above_100mm': np.sum(precipitation > 100),
            
            'max_consecutive_dry_days': self.get_max_consecutive(precipitation == 0),
            'max_consecutive_wet_days': self.get_max_consecutive(precipitation > 0),
        }
    
    def load_checkpoint(self, name):
        """Load model and components from checkpoint."""
        model_path = os.path.join(self.checkpoint_dir, name)
        
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        print(f"Loaded checkpoint from {self.checkpoint_dir}")
    
    def predict(self):
        """Make predictions on test data and create submission file."""
        # Load test data
        test_df = pd.read_csv(os.path.join(self.data_dir, 'Test.csv'))
        
        # Load the scaler
        scaler = joblib.load(os.path.join(self.checkpoint_dir, 'scaler.joblib'))
        
        # Assuming the features are in a DataFrame called test_features
        test_features = test_df.drop(columns=['precipitation'])  # Adjust this line based on your actual data
        
        # Transform the test data
        test_features_scaled = scaler.transform(test_features)
        
        # Make predictions
        dtest = xgb.DMatrix(test_features_scaled)
        predictions = self.model.predict(dtest)
        
        # Create a submission file
        submission_df = pd.DataFrame({
            'id': test_df['id'],  # Adjust this line based on your actual data
            'prediction': predictions
        })
        submission_df.to_csv(os.path.join(self.data_dir, 'submission.csv'), index=False)
        
        print("Predictions saved to submission.csv")

def main():
   
    predictor = FloodPredictor(data_dir='./data', checkpoint_dir='./checkpoints')
    predictor.load_checkpoint("xgboost_model_20241125_122207.json")
    predictor.predict()

if __name__ == "__main__":
    main()