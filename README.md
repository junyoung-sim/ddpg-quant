# Deep Deterministic Policy Gradient and Geometric Brownian Motion for Portfolio Optimization

This algorithm utilizes Geometric Brownian Motion to predict asset valuation cycles fed into a Deep Deterministic Policy Gradient model that maximizes the return of any given portfolio. (WORK IN PROGRESS!!!)

## Geometric Brownian Motion for Estimating Asset Valuation Cycles

Geometric Brownian Motion (GBM) is a stochastic process that models a randomly varying quantity following a Brownian Motion with drift. By simulating stock prices using GBM, this model estimates the probability that the value of a security would be greater than its current value as follows.

![alt text](https://github.com/junyoung-sim/ddpg-quant/blob/main/doc/sample_path.png)
![alt text](https://github.com/junyoung-sim/ddpg-quant/blob/main/doc/price_dist.png)

At a particular time ***t***, the model observes the security's historical data during the past N days and simulate M sample paths for the next K days from ***t***. This yields a lognormal distribution of price values as shown in the second figure above. From that distribution of simulated prices, the model estimates the probability that the security's value would be greater than the current value during the K-day extrapolation period following ***t***. Let this probability be the valuation score (v-score) of the security. By repeating this procedure for every ***t***, the model obtains the following output (valuation series of SPY: S&P 500).

![alt text](https://github.com/junyoung-sim/ddpg-quant/blob/main/doc/vscore.png)

Notice that the critically high or low v-scores and zero-crossings coincide with the extremas and major turning points in the stock index's price movement.

## Deep Deterministic Policy Gradient

The following are some major characteristics of Deep Deterministic Policy Gradients (DDPG):

1) DDPG consists of two components: the actor network and critic network.
2) DDPG can directly map reward-maximizing actions to certain states.
3) DDPG can handle continuous action spaces unlike deep q-networks with discrete action spaces.

### DDPG in a nutshell

The actor network, $\mu(s|\phi)=a$, observes a state from an environment to make a decision where the action space can be continuous. Since the actor must maximize reward, its parameters should be updated by minimizing the following objective function.

$$J=-\log{Q(s,a)}$$

What is $Q(s,a)$? It is the expected reward (q-value), predicted by the the critic network with parameters $\theta$, of performing the action space $a$ given state $s$. Thus, maximizing $Q(s,a)$ is equivalent to minimizing $J$ with respect to $\phi$ via gradient descent to optimize the actor. Simultaneously, we would like the critic network's $Q(s,a)$ to be a high-quality approximation of the true optimal q-value as shown in the Bellman equation below. Note that $r$ is the immediate reward observed after performing $a$ given $s$ and $Q'(s',a')$ is a prediction of future rewards, discounted by $\gamma$, that could be observed by performing $a'$ after state $s'$ that follows $s$.

$$Q^{*}(s,a)=r+{\gamma}Q'(s',a'=\mu'(s'))$$

For stable learning performance, $Q'$ is obtained from a delayed copy (aka target) of the critic network. Similarly, $a'$ is the action space returned by a delayed copy of the actor network given $s'$. Minimizing the following objective function with respect to $\theta$ would improve the critic network's q-value estimations.

$$L=[Q^{*}(s,a)-Q{s,a}]^2$$

Optimizing $J$ and $L$ requires all the techniques used in standard deep Q-learning (e.g., replay memory) with some modifications:

1) We must compute $\frac{dQ}{dA}$ for each action (action gradients) while updating the critic parameters such that we can compute $\frac{dJ}{d\phi}=\frac{dJ}{dQ}\frac{dQ}{dA}\frac{dA}{d\phi}$ for the actor (parallelized via CPU multithreading in this algorithm).
2) We must implement parameter noise, which adds gaussian noise to the actor's parameters to enable exploration while learning. Adding noise to the parameters is proven to be more efficient than adding noise to the action space.

## Portfolio Optimization

Suppose we have a portfolio with N distinct assets and would like to optimize the weights of each asset every market day (really any period) to maximize profit. DDPG is a suitable tool to tackle this problem with the following setup and learning hyperparameters:

**Holdings**
| Ticker | Name |
|--------|------|
| SHY | US 1-3 year Treasury             |
| IEF | US 7-10 year Treasury            |
| HYG | High-Yield Corporate (Junk)      |
| LQD | Investment-Grade Corporate       |
| MBB | Investment-Grade Mortgage-Backed |

**Hyperparameters**
| Parameter | Value |
|-----------|-------|
| Time Period | Mar 2007 - Jan 2024 |
| V-Score Observation Period | 60-days |
| V-Score Extrapolation Period | 20-days |
| V-Score Simulation Epoch | 1000 |
| Look-Back | 100-days (per asset) |
| Iterations | 50 |
| Replay Memory Capacity | 20000 |
| Batch | 10 |
| Initial Epsilon ($\epsilon_0$) | 0.50 |
| Minimum Epsilon ($\epsilon_{min}$) | 0.10 (decay linearly per iteration) |
| Discount Factor ($\gamma$) | 0.90 |
| Learning Rate ($\alpha$) | 0.00000001 |
| L2 Regularization ($\lambda$) | 0.10 |

**Testing in progress. Will post results when available!**

## References

Thanks for teaching me!

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-BM.pdf

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

https://arxiv.org/abs/1509.02971

https://arxiv.org/pdf/2103.11455.pdf

https://spinningup.openai.com/en/latest/algorithms/ddpg.html

https://openai.com/research/better-exploration-with-parameter-noise

... and many others :D