#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include "../lib/quant.hpp"

std::vector<double> sample_state(std::vector<std::vector<double>> &env, unsigned int t) {
    std::vector<double> state;
    for(unsigned int i = 0; i < env.size(); i++) {
        std::vector<double> dat = {env[i].begin() + t+1-OBS, env[i].begin() + t+1};
        state.insert(state.end(), dat.begin(), dat.end());
    }
    return state;
}

std::vector<double> epsilon_greedy(Net &actor, std::vector<double> &state, double eps) {
    double explore = (double)rand() / RAND_MAX;
    std::cout << (explore < eps ? "(E) " : "(P) ");
    return actor.forward(state, explore < eps);
}

void build(std::vector<std::string> &tickers, std::vector<std::vector<double>> &price,
           std::vector<std::vector<double>> &valuation, Net &actor, Net &critic, std::default_random_engine &seed) {
    const unsigned int START = OBS-1;
    const unsigned int TERMINAL = price[0].size()-2;

    unsigned int t0;
    std::random_device rdev;
    std::mt19937 generator(rdev());
    std::uniform_int_distribution<int> dist(START, TERMINAL+1-FRAME);

    std::vector<Memory> replay;

    for(unsigned int itr = 1; itr <= ITR; itr++) {
        t0 = dist(generator);
        double reward_sum = 0.00, reward_mean = 0.00;
        for(unsigned int t = t0; t < t0+FRAME; t++) {
            std::vector<double> state = sample_state(valuation, t);
            std::vector<double> action = epsilon_greedy(actor, state, EPS);

            double reward = 0.00;
            for(unsigned int i = 0; i < tickers.size(); i++) {
                double dp = (price[i][t+1] - price[i][t]) / price[i][t];
                reward += action[i] * (1 + dp);
            }
            reward = (reward - 1.00) * 100;
            reward_sum += reward;
            reward_mean = reward_sum / (t-t0+1);

            std::vector<double> next_state = sample_state(valuation, t+1);

            std::cout << "ITR=" << itr << " T=" << t << " A=[";
            for(unsigned int i = 0; i < tickers.size(); i++) {
                std::cout << tickers[i] << ":" << action[i];
                if(i != tickers.size() - 1) std::cout << ", ";
            }
            std::cout << "] R=" << reward << " MR=" << reward_mean << "\n";

            replay.push_back(Memory(state, action, next_state, reward));

            if(replay.size() == CAPACITY) {
                std::vector<unsigned int> index(CAPACITY, 0);
                std::iota(index.begin(), index.end(), 0);
                std::shuffle(index.begin(), index.end(), seed);
                index.erase(index.begin() + BATCH, index.end());

                //for(unsigned int &k: index) {}

                replay.erase(replay.begin());
            }
        }
    }

    std::vector<Memory>().swap(replay);
}