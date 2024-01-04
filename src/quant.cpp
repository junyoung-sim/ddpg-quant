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
        std::vector<double>().swap(dat);
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
    Net target_actor; copy(actor, target_actor);
    Net target_critic; copy(critic, target_critic);

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

                for(unsigned int &k: index)
                    optimize(replay[k], critic, target_critic, actor, target_actor, ALPHA, LAMBDA);
                
                std::vector<unsigned int>().swap(index);

                replay.erase(replay.begin());
            }
        }

        copy(actor, target_actor);
        copy(critic, target_critic);
    }

    std::vector<Memory>().swap(replay);
}

void optimize(Memory &memory, Net &critic, Net &target_critic,
              Net &actor, Net &target_actor, double alpha, double lambda) {
    std::vector<double> *state = memory.state();
    std::vector<double> *action = memory.action();
    
    std::vector<double> state_action;
    state_action.insert(state_action.end(), action->begin(), action->end());
    state_action.insert(state_action.end(), state->begin(), state->end());

    std::vector<double> *next_state = memory.next_state();
    std::vector<double> next_state_action = target_actor.forward(*next_state, false);
    next_state_action.insert(next_state_action.end(), next_state->begin(), next_state->end());

    std::vector<double> future = target_critic.forward(next_state_action, false);
    double optimal = memory.reward() + GAMMA * future[0];

    unsigned int num_of_tickers = state->size() / OBS;
    std::vector<double> action_gradient(num_of_tickers, 0.00);

    std::vector<double> q = critic.forward(state_action, false);
    for(int l = critic.num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < critic.layer(l)->out_features(); n++) {
            if(l == critic.num_of_layers() - 1) part = -2.00 * (optimal - q[0]);
            else part = critic.layer(l)->node(n)->err() * drelu(critic.layer(l)->node(n)->sum());

            double updated_bias = critic.layer(l)->node(n)->bias() - alpha * part;
            critic.layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < critic.layer(l)->in_features(); i++) {
                if(l == 0) {
                    grad = part * state_action[i];
                    if(i < num_of_tickers)
                        action_gradient[i] = part * critic.layer(l)->node(n)->weight(i);
                }
                else {
                    grad = part * critic.layer(l-1)->node(i)->act();
                    critic.layer(l-1)->node(i)->add_err(part * critic.layer(l)->node(n)->weight(i));
                }

                grad += lambda * critic.layer(l)->node(n)->weight(i);

                double updated_weight = critic.layer(l)->node(n)->weight(i) - alpha * grad;
                critic.layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}