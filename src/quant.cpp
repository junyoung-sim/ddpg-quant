#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <thread>
#include <fstream>

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

    std::vector<Memory> replay;

    std::ofstream out("./res/build");
    out << "total_return\n";

    for(unsigned int itr = 1; itr <= ITR; itr++) {
        double total_return = 1.00;
        double eps = (EPS_MIN - EPS_INIT) / (ITR-1) * (itr-1) + EPS_INIT;
        for(unsigned int t = START; t <= TERMINAL; t++) {
            std::vector<double> state = sample_state(valuation, t);
            std::vector<double> action = epsilon_greedy(actor, state, eps);

            double reward = 0.00;
            for(unsigned int i = 0; i < tickers.size(); i++) {
                double dp = (price[i][t+1] - price[i][t]) / price[i][t];
                reward += action[i] * (1.00 + dp);
            }
            reward = (reward - 1.00) * 100;
            total_return *= 1.00 + reward / 100;

            std::vector<double> next_state = sample_state(valuation, t+1);

            std::cout << "ITR=" << itr << " T=" << t << " A=[";
            for(unsigned int i = 0; i < tickers.size(); i++) {
                std::cout << tickers[i] << ":" << action[i];
                if(i != tickers.size() - 1) std::cout << ", ";
            }
            std::cout << "] R=" << reward << " TR=" << total_return << "\n";

            replay.push_back(Memory(state, action, next_state, reward));

            if(replay.size() == CAPACITY) {
                std::vector<unsigned int> index(replay.size(), 0);
                std::iota(index.begin(), index.end(), 0);
                std::shuffle(index.begin(), index.end(), seed);
                index.erase(index.begin() + BATCH, index.end());

                for(unsigned int &k: index)
                    optimize(replay[k], critic, target_critic, actor, target_actor, ALPHA, LAMBDA);
                
                std::vector<unsigned int>().swap(index);

                replay.erase(replay.begin());
            }
        }

        out << total_return << "\n";

        copy(actor, target_actor);
        copy(critic, target_critic);
    }

    out.close();

    std::vector<Memory>().swap(replay);
}

void optimize_critic(Net &critic, std::vector<double> &state_action, std::vector<double> &agrad,
                     std::vector<bool> &flag, double optimal, double q, double alpha, double lambda, unsigned int num_of_tickers) {
    for(int l = critic.num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < critic.layer(l)->out_features(); n++) {
            if(l == critic.num_of_layers() - 1) part = -2.00 * (optimal - q);
            else part = critic.layer(l)->node(n)->err() * drelu(critic.layer(l)->node(n)->sum());

            double updated_bias = critic.layer(l)->node(n)->bias() - alpha * part;
            critic.layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < critic.layer(l)->in_features(); i++) {
                if(l == 0) {
                    grad = part * state_action[i];
                    if(i < num_of_tickers) {
                        agrad[i] = part * critic.layer(l)->node(n)->weight(i);
                        flag[i] = true;
                    }
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

void optimize_actor(Net &actor, std::vector<double> &state, std::vector<double> &action,
                    std::vector<double> &agrad, std::vector<bool> &flag, double q, double alpha, double lambda) {
    for(int l = actor.num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < actor.layer(l)->out_features(); n++) {
            if(l == actor.num_of_layers() - 1) {
                while(!flag[n]);
                part = -1.00 / q * agrad[n] * action[n] * (1.00 - action[n]);
            }
            else part = actor.layer(l)->node(n)->err() * drelu(actor.layer(l)->node(n)->sum());

            double updated_bias = actor.layer(l)->node(n)->bias() - alpha * part;
            actor.layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < actor.layer(l)->in_features(); i++) {
                if(l == 0) grad = part * state[i];
                else {
                    grad = part * actor.layer(l-1)->node(i)->act();
                    actor.layer(l-1)->node(i)->add_err(part * actor.layer(l)->node(n)->weight(i));
                }

                grad += lambda * actor.layer(l)->node(n)->weight(i);

                double updated_weight = actor.layer(l)->node(n)->weight(i) - alpha * grad;
                actor.layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

void optimize(Memory &memory, Net &critic, Net &target_critic,
              Net &actor, Net &target_actor, double alpha, double lambda) {
    std::vector<double> *state = memory.state();
    std::vector<double> *action = memory.action();

    std::vector<double> state_action;
    state_action.insert(state_action.end(), action->begin(), action->end());
    state_action.insert(state_action.end(), state->begin(), state->end());

    std::vector<double> q = critic.forward(state_action, false);

    std::vector<double> *next_state = memory.next_state();
    std::vector<double> next_state_action = target_actor.forward(*next_state, false);
    next_state_action.insert(next_state_action.end(), next_state->begin(), next_state->end());

    std::vector<double> future = target_critic.forward(next_state_action, false);
    double optimal = memory.reward() + GAMMA * future[0];

    unsigned int num_of_tickers = state->size() / OBS;
    std::vector<double> agrad(num_of_tickers, 0.00);
    std::vector<bool> flag(num_of_tickers, false);

    std::thread critic_optimizer(optimize_critic, std::ref(critic), std::ref(state_action),
                                 std::ref(agrad), std::ref(flag), optimal, q[0], ALPHA, LAMBDA, num_of_tickers);

    std::thread actor_optimizer(optimize_actor, std::ref(actor), std::ref(*state),
                                std::ref(*action), std::ref(agrad), std::ref(flag), q[0], ALPHA, LAMBDA);

    critic_optimizer.join();
    actor_optimizer.join();

    std::vector<double>().swap(state_action);
    std::vector<double>().swap(q);
    std::vector<double>().swap(next_state_action);
    std::vector<double>().swap(future);
    std::vector<double>().swap(agrad);
    std::vector<bool>().swap(flag);
}