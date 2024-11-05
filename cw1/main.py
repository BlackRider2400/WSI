import matplotlib.pyplot as plt

# Check Matplotlib installation
try:
    print("Matplotlib version:", plt.__version__)
except Exception as e:
    print("Error importing Matplotlib:", e)

# Simple subplot test
try:
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Data for plotting
    x = [1, 2, 3, 4, 5]
    y1 = [2, 4, 6, 8, 10]
    y2 = [1, 3, 5, 7, 9]
    y3 = [5, 3, 2, 6, 4]
    y4 = [7, 8, 5, 3, 6]

    # Plot each subplot
    axs[0, 0].plot(x, y1, label="y = 2x", color="blue")
    axs[0, 0].set_title("Top Left")
    axs[0, 0].legend()

    axs[0, 1].plot(x, y2, label="y = 2x - 1", color="green")
    axs[0, 1].set_title("Top Right")
    axs[0, 1].legend()

    axs[1, 0].plot(x, y3, label="Random Plot 1", color="red")
    axs[1, 0].set_title("Bottom Left")
    axs[1, 0].legend()

    axs[1, 1].plot(x, y4, label="Random Plot 2", color="purple")
    axs[1, 1].set_title("Bottom Right")
    axs[1, 1].legend()

    # Display the subplots
    plt.tight_layout()
    plt.show()
except Exception as e:
    print("Error plotting subplots with Matplotlib:", e)
