#!/usr/bin/env python3

import math as math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def annualized_return(value):
    n = 0
    expectation = 1.00
    returns = []
    for t0 in range(0, value.shape[0], 251):
        tf = min(t0 + 251, value.shape[0] - 1)
        expectation *= 1.00 + (value[tf] - value[t0]) / value[t0]
        returns.append(expectation)
        n += 1
    annualized = math.pow(expectation, 1 / n) - 1.00
    return annualized, np.array(returns)

def maximum_drawdown(value):
    mdd = 0.00
    dp = np.copy(value)
    for t in range(1, value.shape[0]):
        dp[t] = max(dp[t-1], dp[t])
        mdd = min((value[t] - dp[t]) / dp[t], mdd)
    return abs(mdd)

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

    apr, returns = annualized_return(portfolio["value"])
    stdev = np.std(returns)
    sharpe = apr / stdev
    mdd = maximum_drawdown(portfolio["value"])

    print(f"APR={apr}")
    print(f"STD={stdev}")
    print(f"SHR={sharpe}")
    print(f"MDD={mdd}")