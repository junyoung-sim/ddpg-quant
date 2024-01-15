#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt

log = pd.read_csv("./res/build")

plt.figure(figsize=(10,5))
plt.plot(log["total_return"])
plt.savefig("./res/build.png")