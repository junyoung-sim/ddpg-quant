#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WINDOW = 10

def mavg(x, w):
    return np.convolve(x, np.ones(w), "valid") / w

if __name__ == "__main__":
    log = pd.read_csv("./res/build")
    log["return"] = (log["return"] - 1.00) * 100

    plt.figure(figsize=(10,5))

    plt.subplot(3, 1, 1)
    plt.plot(log["return"], label="return (%)")
    return_mavg = mavg(log["return"], WINDOW)
    for i in range(WINDOW-1):
        return_mavg = np.insert(return_mavg, 0, np.nan)
    plt.plot(return_mavg, color="orange")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(log["sharpe"], label="sharpe")
    sharpe_mavg = mavg(log["sharpe"], WINDOW)
    for i in range(WINDOW-1):
        sharpe_mavg = np.insert(sharpe_mavg, 0, np.nan)
    plt.plot(sharpe_mavg, color="orange")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(log["loss"], label="loss")
    plt.legend()

    plt.savefig("./res/build.png")