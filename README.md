# Deep Deterministic Policy Gradient and Geometric Brownian Motion for Portfolio Optimization

This algorithm utilizes Geometric Brownian Motion to predict asset valuation cycles fed into a Deep Deterministic Policy Gradient model that maximizes the return of any given portfolio.

## Geometric Brownian Motion for Estimating Asset Valuation Cycles

Geometric Brownian Motion (GBM) is a stochastic process that models a randomly varying quantity following a Brownian Motion with drift. By simulating stock prices using GBM, this model estimates the probability that the value of a security would be greater than its current value as follows.

![alt text](https://github.com/junyoung-sim/ddpg-quant/blob/main/doc/sample_path.png)
![alt text](https://github.com/junyoung-sim/ddpg-quant/blob/main/doc/price_dist.png)

At a particular time ***t***, the model observes the security's historical data during the past N days and simulate M sample paths for the next K days from ***t***. This yields a lognormal distribution of price values as shown in the second figure above. From that distribution of simulated prices, the model estimates the probability that the security's value would be greater than the current value during the K-day extrapolation period following ***t***. Let this probability be the valuation score (v-score) of the security. By repeating this procedure for every ***t***, the model obtains the following output (valuation series of SPY: S&P 500).

![alt text](https://github.com/junyoung-sim/ddpg-quant/blob/main/doc/vscore.png)

Notice that the critically high or low v-scores and zero-crossings coincide with the extremas and major turning points in the stock index's price movement.

## Portfolio Optimization with Deep Deterministic Policy Gradient and GBM

Suppose we have a portfolio with N distinct assets and would like to optimize the weights of each asset every market day (really any period) to maximize profit. Deep Deterministic Policy Gradients (DDPG) can be a useful technique to tackle such a problem because of the following characteristics:

1) DDPG is consists of two components: the actor network and critic network.
2) DDPG can directly map reward-maximizing actions to certain states.
3) DDPG can handle continuous action spaces unlike deep q-networks with discrete action spaces.

### Actor and Critic in DDPG

The actor network ($\mu(s|\phi)=a$) observes a state from an environment to make a reward-maximizing decision. Note that a softmax layer can be used to implement a continuous action space. The actor's parameters should be updated via the following objective function.

$$J=-\log{Q(s,a)}$$

What is $Q(s,a)$? It is the expected reward (q-value), predicted by the the critic network with parameters $\theta$, of performing the action space $a$ given state $s$. Since we would like to maximize reward, we must maximize $Q(s,a)$, thus minimizing $J$ with respect to $\phi$ via gradient descent would optimize the actor's behavior. Simultaneously, we would like the critic network's $Q(s,a)$ to be a high-quality approximation of the true optimal q-value as shown in the Bellman equation below. Note that $r$ is the immediate reward observed after performing $a$ given $s$ and $Q'(s',a')$ is a prediction of future rewards, discounted by $\gamma$, that could be observed by performing $a'$ after state $s'$ that follows $s$.

$$Q^{*}(s,a)=r+{\gamma}Q'(s',a')$$

For stable learning performance, $Q'$ is obatined from a delayed copy (aka target) of the critic network. Similarly, $a'$ is the action space returned by a delayed copy of the actor network given $s'$. Minimizing the following objective function with respect to $\theta$ would improve the critic network's q-value estimations.

$$L=[Q^{*}(s,a)-Q{s,a}]^2$$

Optimizing $J$ and $L$ requires all the techniques used in standard deep Q-learning (e.g., replay memory) with some modifications:

1) We must compute $\frac{dQ}{dA}$ for each action (action gradients) while updating the critic parameters such that we can compute $\frac{dJ}{d\phi}=\frac{dJ}{dQ}\frac{dQ}{dA}\frac{dA}{d\phi}$ for the actor (parallelized via CPU multithreading in this algorithm).
2) We must implement parameter noise, which adds gaussian noise to the actor's parameters to enable exploration while learning. Adding noise to the parameters is proven to be more efficient than adding noise to the action space.

### Setup & Results

In this application, let the state space be the valuation series of N assets in a given portfolio and the action space be the weights allocated for each asset. The N assets could be anything, so let's build a model for optimizing a bond portfolio (SHY: 1-3 yr UST, IEF: 10 yr UST, TLT: 20 yr UST, HYG: high-yield junk, LQD: investment grade). The following are some hyperparameters for training:

1) Historical Period: 2002 - 2023
2) Iterations: 100 (traverse entire historical period per iteration)
3) V-Score Observation Period: 60-days
4) V-Score Extrapolation Period: 20-days
5) V-Score Simulation Epochs: 1000
6) State Space Look-Back: 100-days (per asset)
7) Batch: 10 experiences
8) Initial Epsilon: 1.00 (probability of adding parameter noise)
9) Minimum Epsilon: 0.10 (decay linearly per iteration)
10) Gamma: 0.90
11) Learning Rate: 0.00000001
12) L2 Regularization: 0.10

**Testing in progress. Looking good so far, will post results when available!**

## References

Thanks for teaching me!

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-BM.pdf

http://www.columbia.edu/~ks20/FE-Notes/4700-07-Notes-GBM.pdf

https://arxiv.org/abs/1509.02971

https://arxiv.org/pdf/2103.11455.pdf

https://spinningup.openai.com/en/latest/algorithms/ddpg.html

https://openai.com/research/better-exploration-with-parameter-noise

... and many others :D