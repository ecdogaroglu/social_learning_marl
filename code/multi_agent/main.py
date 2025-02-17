from ctde import train_ctde
from dtde import train_dtde
from metrics import plot_multi_agent_metrics
from pathlib import Path

# Current script path
current_path = Path(__file__).resolve()

# Move to the parent directory and specify the new folder
save_folder = current_path.parent.parent.parent / "charts"
save_folder.mkdir(exist_ok=True)  # Create folder if it doesn't exist


if __name__ == "__main__":
    # Train with metrics tracking
    num_agents = 4
    num_steps = 1000
    signal_accuracies = [0.75]

    for signal_accuracy in signal_accuracies:

        # Train models
        trained_model_ctde, metrics_ctde = train_ctde(num_agents=num_agents, num_steps=num_steps, signal_accuracy=signal_accuracy)
        trained_model_dtde, metrics_dtde = train_dtde(num_agents=num_agents, num_steps=num_steps, signal_accuracy=signal_accuracy)

        # Plot results
        plot_multi_agent_metrics(metrics_ctde, signal_accuracy=signal_accuracy, save_path=save_folder/f"multi_agent_ctde_learning_curves_q={signal_accuracy}.png")
        plot_multi_agent_metrics(metrics_dtde, signal_accuracy=signal_accuracy, save_path=save_folder/f"multi_agent_dtde_learning_curves_q={signal_accuracy}.png")



