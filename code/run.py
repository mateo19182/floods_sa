import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import get_prediction
from train import train_and_evaluate
from data import BASE_PATH
import wandb
from data import get_datasets

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def visualize_results(final_state, valid_labels, losses):
    event_id = 0
    x = np.arange(len(losses[event_id, :]))
    plt.title('Validation set results')
    plt.plot(x, losses[event_id, :], label='Predicted losses of flood', color='blue')
    plt.plot(x, valid_labels[event_id, :], label='Label for flooding event', color='red')
    plt.ylabel('Losses value for flood probability')
    plt.xlabel('event_t')
    plt.legend()
    plt.show()

def create_submission(final_state, test_ds):
    new_model_state, logits = get_prediction(final_state, test_ds, is_training=False)

    probs = sigmoid(logits)
    sample_submission = pd.read_csv(BASE_PATH + '/SampleSubmission.csv')
    sample_submission['label'] = probs.flatten()
    sample_submission.to_csv('BenchmarkSubmission.csv', index=False)

def main():
    # Initialize wandb for logging
    wandb.init(project="floods")

    # Hyperparameters
    num_epochs = 150
    batch_size = 64
    use_images = True

    # Train and evaluate the model
    final_state, losses = train_and_evaluate(
        num_epochs=num_epochs,
        batch_size=batch_size,
        use_images=use_images,
    )

    # Get datasets for visualization and submission
    _, _, test_ds, _ = get_datasets()
    test_inputs = (test_ds['timeseries'], test_ds['image'])

    # Visualize validation set results
    # visualize_results(final_state, valid_ds['label'], losses)

    # Create submission file
    create_submission(final_state, test_inputs)

if __name__ == "__main__":
    main()