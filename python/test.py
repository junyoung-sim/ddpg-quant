#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    state = pd.read_csv("./res/state")
    action = pd.read_csv("./res/action")
    portfolio = pd.read_csv("./res/portfolio")

    colors = ["cadetblue", "steelblue", "dodgerblue", "lightskyblue"]

    k = 1
    plt.figure(figsize=(25,15))
    for key in state.keys():
        plt.subplot(state.keys().shape[0], 1, k)
        plt.title(f"State: {key}")
        plt.plot(state[key], label=key, color=colors[k-1])
        plt.legend()
        k += 1
    plt.savefig("./res/test_state.png")

    ###

    plt.figure(figsize=(25,15))
    plt.subplot(2, 1, 1)
    plt.title("Portfolio Value")
    plt.plot(portfolio["value"], label="Portfolio", color="seagreen")
    plt.legend()

    k = 1
    plt.subplot(2, 1, 2)
    plt.title("Portfolio Weights")
    for key in action.keys():
        plt.plot(action[key], label=key, color=colors[k-1])
        plt.legend()
        k += 1
    plt.ylim(0.00, 0.70)

    plt.savefig("./res/test_portfolio.png")