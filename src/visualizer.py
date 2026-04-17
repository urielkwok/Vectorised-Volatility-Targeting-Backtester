import matplotlib.pyplot as plt


def create_plots(results):
    total_base_returns = (results["total_base_returns"] - 1) * 100
    total_vol_returns = (results["total_vol_returns"] - 1) * 100
    plt.title("Returns vs Time")
    plt.xlabel("Time")
    plt.ylabel("Returns (%)")
    plt.plot(total_base_returns, label="base returns")
    plt.plot(total_vol_returns, label="strategy returns")
    plt.legend()
    plt.tight_layout()
    plt.show()
