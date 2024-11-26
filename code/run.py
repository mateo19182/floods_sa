import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from train import get_prediction, load_model, train_and_evaluate
from data import BASE_PATH
import wandb
from data import get_datasets

def sigmoid(x):
    x = np.clip(x, -700, 700)
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
    print("Submission file created successfully at BenchmarkSubmission.csv")

def main():
    # # Initialize wandb for logging
    wandb.init(project="floods")

    train_ds, valid_ds, test_ds, seq_length = get_datasets(
        augment=True
    )

    # Train and evaluate the model
    final_state, losses = train_and_evaluate(
        num_epochs=250,
        batch_size=32,
        # learning_rate=1e-3,
        use_images=True,
        train_ds=train_ds,
        valid_ds=valid_ds, 
        seq_length=seq_length
    )

    # Load the best model
    # _, _, test_ds, _ = get_datasets(
    #      augment=False
    # )
    # final_state = load_model("/home/mateo/projects/ai/comps/floods_sa/code/checkpoints/best_f1_score")

    # Visualize validation set results
    # visualize_results(final_state, valid_ds['label'], losses)

    # Create submission file
    create_submission(final_state, (test_ds['timeseries'], test_ds['image']))

if __name__ == "__main__":
    main()