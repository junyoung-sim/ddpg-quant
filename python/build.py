#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WINDOW = 10

def mavg(x, w):
    return np.convolve(x, np.ones(w), "valid") / w

if __name__ == "__main__":
    log = pd.read_csv("./res/build")
    log = log.dropna().reset_index()

    plt.figure(figsize=(10,5))

    plt.subplot(3, 1, 1)
    plt.plot(log["total"], label="total (%)")
    total_mavg = mavg(log["total"], WINDOW)
    for i in range(WINDOW-1):
        total_mavg = np.insert(total_mavg, 0, np.nan)
    plt.plot(total_mavg, color="orange")
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(log["daily"], label="daily (%)")
    daily_mavg = mavg(log["daily"], WINDOW)
    for i in range(WINDOW-1):
        daily_mavg = np.insert(daily_mavg, 0, np.nan)
    plt.plot(daily_mavg, color="orange")
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(log["loss"], label="loss")
    plt.legend()

    plt.savefig("./res/build.png")