#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("./res/build")

plt.figure(figsize=(10,5))

plt.subplot(3, 1, 1)
plt.plot(log["return"], label="return")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(log["sharpe"], label="sharpe")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(log["loss"], label="loss")
plt.legend()

plt.savefig("./res/build.png")