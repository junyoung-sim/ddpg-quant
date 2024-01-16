#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("./res/build")

plt.figure(figsize=(10,5))

plt.subplot(2, 1, 1)
plt.plot(log["total_return"])

plt.subplot(2, 1, 2)
plt.plot(log["actor_loss"])

plt.savefig("./res/build.png")