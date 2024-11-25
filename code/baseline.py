import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import os
from datetime import datetime
from tqdm import tqdm

class FloodPredictor:
    def __init__(self, data_dir='./data', checkpoint_dir='./checkpoints'):
        self.data_dir = data_dir
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.min_values = None
        self.max_values = None

    def create_features(self, df):
        """Generate features for the dataset, including time-related features."""
        print("Creating features...")

        features = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            row_dict = {'event_id': row['event_id']}
            precipitation = row['precipitation']
            event_id_parts = row['event_id'].split('_')  # Split event_id_X_day into parts
            day = event_id_parts[-1]  # Extract the day part

            day_in_event = int(day)  # Extract the day in the event sequence
            
            # Rolling window of precipitation (1-day, 3-day, 7-day windows)
            rolling_1d_mean = row['precipitation'] if _ == 0 else df.loc[_-1:_]['precipitation'].mean()
            rolling_3d_mean = row['precipitation'] if _ < 2 else df.loc[_-2:_]['precipitation'].mean()
            rolling_7d_mean = row['precipitation'] if _ < 6 else df.loc[_-6:_]['precipitation'].mean()

            # Basic precipitation statistics for each row (not grouped by base_id)
            row_dict.update({
                'precip_mean': precipitation,
                'precip_max': precipitation,
                'precip_sum': precipitation,
                'precip_zeros': 1 if precipitation == 0 else 0,  # Zero precipitation flag
                'rolling_1d_mean': rolling_1d_mean,
                'rolling_3d_mean': rolling_3d_mean,
                'rolling_7d_mean': rolling_7d_mean,
                'day_in_event': day_in_event,  # Day within the event
            })

            # If label exists, use it
            if 'label' in row:
                row_dict['label'] = row['label']

            features.append(row_dict)

        return pd.DataFrame(features)

    def normalize_features(self, df, train=True):
        """Normalize features using min-max scaling."""
        if train:
            self.min_values = df.min()
            self.max_values = df.max()
        
        return (df - self.min_values) / (self.max_values - self.min_values)

    def load_and_preprocess(self):
        """Load and preprocess the data with feature engineering."""
        print("Loading data...")
        self.train_df = pd.read_csv(f"{self.data_dir}/Train.csv")
        self.test_df = pd.read_csv(f"{self.data_dir}/Test.csv")

        print("Generating training features...")
        features_df = self.create_features(self.train_df)
        self.X = features_df.drop(['label', 'event_id'], axis=1)
        self.y = features_df['label']

        # Normalize features
        self.X = self.normalize_features(self.X, train=True)

        # Handle class imbalance
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(self.y), y=self.y)
        self.class_weight_dict = dict(zip(np.unique(self.y), class_weights))

        # Split data into training and validation sets
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )

        # Create DMatrix objects for XGBoost
        self.dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        self.dval = xgb.DMatrix(self.X_val, label=self.y_val)

    def preprocess_test_data(self, test_df):
        """Preprocess the test data."""
        print("Generating test features...")
        test_features = self.create_features(test_df)

        # Drop the 'event_id' column (it is not a feature)
        X_test = test_features.drop(['event_id'], axis=1, errors='ignore')

        # Normalize using the training normalization
        X_test = self.normalize_features(X_test, train=False)

        # Return DMatrix with correct feature names
        return xgb.DMatrix(X_test)

    def train_model(self):
        """Train the XGBoost model."""
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'booster': 'gbtree',
            'verbosity': 0,
            'n_estimators': 1000,
            'learning_rate': 0.01,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': self.class_weight_dict[1],  # Handling class imbalance
        }

        # Train model
        self.model = xgb.train(params, self.dtrain, num_boost_round=1000, evals=[(self.dval, 'eval')], early_stopping_rounds=50)
        
        # Save the model
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        best_eval = self.model.best_score
        checkpoint_path = os.path.join(self.checkpoint_dir, f'best_model_{timestamp}_eval{best_eval:.4f}.bin')
        self.model.save_model(checkpoint_path)
        print(f"Model saved to {checkpoint_path}")
    
    def generate_predictions(self):
        """Generate predictions for the test dataset."""
        print("\nProcessing test data for predictions...")

        # Preprocess the test data
        dtest = self.preprocess_test_data(self.test_df)

        # Generate predictions
        test_predictions = self.model.predict(dtest)

        # Ensure the predictions match the original test data length
        if len(test_predictions) != len(self.test_df):
            raise ValueError(
                f"Prediction length ({len(test_predictions)}) does not match test data length ({len(self.test_df)})"
            )

        # Save predictions in the required format
        self.test_df['prediction'] = test_predictions
        output_file = f"./predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.test_df[['event_id', 'prediction']].to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

def main():
    # Initialize FloodPredictor
    predictor = FloodPredictor(data_dir='./data', checkpoint_dir='./checkpoints')

    # Load and preprocess training data
    predictor.load_and_preprocess()

    # Train the model
    print("\nTraining the model...")
    predictor.train_model()

    # Generate predictions on test data
    print("\nGenerating predictions...")
    predictor.generate_predictions()

# Run the script
if __name__ == "__main__":
    main()
