import matplotlib.pyplot as plt
import seaborn as sns

def plot_stats():
    # Data for the plot
    stats = {
        "DR": [5, 8.33],
        "Memory (MB)": [376.60, 345.45],
        "BER (%) - ASK": [4, 24],
        "Data Rate (bps)": [5, 8.33]
    }
    
    labels = ["ASK", "CSS"]

    # Set Seaborn style for a modern, minimalist look
    sns.set_theme(style="whitegrid", palette="muted")
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(4, 4))
    axes = axes.flatten()

    # Plot CPU usage with minimalist design
    axes[0].bar(labels, stats["DR"], color='steelblue', width=0.4)
    axes[0].set_title("Data Rate", fontsize=14, weight='bold')
    axes[0].set_ylabel("bps", fontsize=12)
    axes[0].tick_params(axis='both', which='major', labelsize=11)
    
    # Plot Memory usage with minimalist design
    axes[1].bar(labels, stats["Memory (MB)"], color='seagreen', width=0.4)
    axes[1].set_title("Memory Usage (MB)", fontsize=14, weight='bold')
    axes[1].set_ylabel("Memory (MB)", fontsize=12)
    axes[1].tick_params(axis='both', which='major', labelsize=11)

    # Plot BER for ASK and CSS with minimalist design
    axes[2].bar(labels, stats["BER (%) - ASK"], color='salmon', width=0.4)
    axes[2].set_title("Bit Error Rate (BER) (%)", fontsize=14, weight='bold')
    axes[2].set_ylabel("BER (%)", fontsize=12)
    axes[2].tick_params(axis='both', which='major', labelsize=11)

    # Plot Data Rate for ASK and CSS with minimalist design
    axes[3].bar(labels, stats["Data Rate (bps)"], color='dodgerblue', width=0.4)
    axes[3].set_title("Data Rate (bps)", fontsize=14, weight='bold')
    axes[3].set_ylabel("Data Rate (bps)", fontsize=12)
    axes[3].tick_params(axis='both', which='major', labelsize=11)

    # Adjust layout and remove unnecessary spines
    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.yaxis.set_tick_params(width=0.8)
        ax.xaxis.set_tick_params(width=0.8)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

# Call the function to generate the plots
plot_stats()
