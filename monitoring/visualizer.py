import matplotlib.pyplot as plt

def plot_reconstruction_error(errors, drift_points=None):
    plt.figure(figsize=(12, 6))
    plt.plot(errors, label="Reconstruction Error")
    if drift_points:
        for point in drift_points:
            plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
    plt.title("Reconstruction Error Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.show()
