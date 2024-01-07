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

1) DDPG is an actor-critic model.
2) DDPG can handle continuous action spaces (unlike deep q-networks with discrete action spaces).
2) DDPG can directly map reward-maximizing actions to certain states.

### Actor and Critic in DDPG

The actor network observes a state from the environment to make a reward-maximizing decision. In this application, the actor network observes the valuation series (n=60, m=1000, k=20) of N assets in a given portfolio and outputs the weights allocated for each asset via a softmax layer. The actor's parameters should be updated via the following objective function.

$$J=-logQ(s,a)$$