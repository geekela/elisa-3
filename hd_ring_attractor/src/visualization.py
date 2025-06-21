import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_activity(activity, title="Neural Activity"):
    plt.figure(figsize=(10, 6))
    plt.imshow(activity.T, aspect='auto', cmap='hot', origin='lower')
    plt.colorbar(label='Activity')
    plt.xlabel('Time step')
    plt.ylabel('Neuron')
    plt.title(title)
    return plt.gcf()


def plot_weights(weights, title="Connection Weights"):
    plt.figure(figsize=(8, 8))
    plt.imshow(weights, cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(label='Weight')
    plt.xlabel('From neuron')
    plt.ylabel('To neuron')
    plt.title(title)
    return plt.gcf()


def plot_trajectory(true_angles, predicted_angles=None, title="Head Direction Trajectory"):
    plt.figure(figsize=(12, 6))
    
    time_steps = np.arange(len(true_angles))
    
    # Convert angles to degrees
    true_angles_deg = np.degrees(true_angles)
    if predicted_angles is not None:
        predicted_angles_deg = np.degrees(predicted_angles)
    
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, true_angles_deg, 'b-', label='True angle')
    if predicted_angles is not None:
        plt.plot(time_steps, predicted_angles_deg, 'r--', label='Predicted angle')
    plt.xlabel('Time step')
    plt.ylabel('Angle (degrees)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.polar(true_angles, time_steps, 'b-', label='True')
    if predicted_angles is not None:
        plt.polar(predicted_angles, time_steps, 'r--', label='Predicted')
    plt.title('Polar representation')
    
    plt.tight_layout()
    return plt.gcf()


def plot_tuning_curves(model, n_samples=100):
    # Get the device from the model parameters
    device = next(model.parameters()).device
    angles = torch.linspace(0, 2 * np.pi, n_samples, device=device)
    activities = []
    
    for angle in angles:
        x = torch.tensor([[torch.cos(angle), torch.sin(angle)]], device=device)
        h = model(x, steps=50)
        activities.append(h.squeeze().detach().numpy())
    
    activities = np.array(activities)
    
    # Convert angles to degrees for plotting
    angles_deg = np.degrees(angles.numpy())
    
    plt.figure(figsize=(12, 8))
    n_neurons_to_plot = min(10, model.n_neurons)
    neuron_indices = np.linspace(0, model.n_neurons-1, n_neurons_to_plot, dtype=int)
    
    for i, neuron_idx in enumerate(neuron_indices):
        plt.plot(angles_deg, activities[:, neuron_idx], label=f'Neuron {neuron_idx}')
    
    plt.xlabel('Input angle (degrees)')
    plt.ylabel('Activity')
    plt.title('Tuning curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    
    return plt.gcf()