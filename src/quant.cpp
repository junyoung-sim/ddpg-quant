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
    for(std::vector<double> &dat: env) {
        for(unsigned int i = t-OBS+INT; i <= t; i += INT)
            state.push_back(dat[i]);
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
    Net target_actor; copy(actor, target_actor, 1.00);
    Net target_critic; copy(critic, target_critic, 1.00);

    const unsigned int START = OBS-1;
    const unsigned int TERMINAL = price[0].size()-2;
    unsigned int frames = 0;

    std::vector<Memory> replay;

    std::ofstream out("./res/build");
    out << "total_return,actor_loss\n";

    for(unsigned int itr = 1; itr <= ITR; itr++) {
        double total_return = 1.00;
        double actor_loss_sum = 0.00;
        unsigned int update_count = 0;
        for(unsigned int t = START; t <= TERMINAL; t++) {
            double eps = std::max(EPS_MIN, EPS_INIT + (EPS_MIN - EPS_INIT) / CAPACITY * frames++);
            std::vector<double> state = sample_state(valuation, t);
            std::vector<double> action = epsilon_greedy(actor, state, eps);

            double portfolio_return = 0.00, portfolio_risk = 0.00, sharpe = 0.00;
            for(unsigned int i = 0; i < tickers.size(); i++) {
                double dp = (price[i][t+1] - price[i][t]) / price[i][t];
                portfolio_return += action[i] * (1.00 + dp);

                std::vector<double> tmp = {price[i].begin(), price[i].begin() + t+1};
                std::vector<double> r = returns(tmp);
                portfolio_risk += action[i] * pow(stdev(r), 2);
            }
            total_return *= portfolio_return;
            sharpe = std::max(0.00, (portfolio_return - 1.00) / sqrt(portfolio_risk));

            std::vector<double> next_state = sample_state(valuation, t+1);

            std::cout << "ITR=" << itr << " T=" << t << " A=[";
            for(unsigned int i = 0; i < tickers.size(); i++) {
                std::cout << tickers[i] << ":" << action[i];
                if(i != tickers.size() - 1) std::cout << ", ";
            }
            std::cout << "] TR=" << total_return << " SR=" << sharpe;
            std::cout << " L=" << actor_loss_sum / update_count << "\n";

            replay.push_back(Memory(state, action, next_state, sharpe));

            if(replay.size() == CAPACITY) {
                std::vector<unsigned int> index(CAPACITY, 0);
                std::iota(index.begin(), index.end(), 0);
                std::shuffle(index.begin(), index.end(), seed);
                index.erase(index.begin() + BATCH, index.end());

                for(unsigned int &k: index)
                    actor_loss_sum += optimize(replay[k], critic, target_critic, actor, target_actor);
                update_count += BATCH;
                
                copy(actor, target_actor, TAU);
                copy(critic, target_critic, TAU);

                replay.erase(replay.begin());

                std::vector<unsigned int>().swap(index);
            }
        }

        out << total_return << "," << actor_loss_sum / update_count << "\n";
    }

    out.close();
    std::system("./python/build.py");

    std::vector<Memory>().swap(replay);
}

void optimize_critic(Net &critic, std::vector<double> &state_action, double optimal, double q,
                     unsigned int num_of_tickers, std::vector<double> &agrad, std::vector<bool> &flag) {
    for(int l = critic.num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < critic.layer(l)->out_features(); n++) {
            if(l == critic.num_of_layers() - 1) part = -2.00 * (optimal - q);
            else part = critic.layer(l)->node(n)->err() * drelu(critic.layer(l)->node(n)->sum());

            double updated_bias = critic.layer(l)->node(n)->bias() - ALPHA * part;
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

                grad += LAMBDA * critic.layer(l)->node(n)->weight(i);

                double updated_weight = critic.layer(l)->node(n)->weight(i) - ALPHA * grad;
                critic.layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

void optimize_actor(Net &actor, std::vector<double> &state,
                    std::vector<double> &action, std::vector<double> &agrad, std::vector<bool> &flag) {
    for(int l = actor.num_of_layers() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < actor.layer(l)->out_features(); n++) {
            if(l == actor.num_of_layers() - 1) {
                while(!flag[n]);
                part = agrad[n] * action[n] * (1.00 - action[n]);
            }
            else part = actor.layer(l)->node(n)->err() * drelu(actor.layer(l)->node(n)->sum());

            double updated_bias = actor.layer(l)->node(n)->bias() - ALPHA * part;
            actor.layer(l)->node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < actor.layer(l)->in_features(); i++) {
                if(l == 0) grad = part * state[i];
                else {
                    grad = part * actor.layer(l-1)->node(i)->act();
                    actor.layer(l-1)->node(i)->add_err(part * actor.layer(l)->node(n)->weight(i));
                }

                grad += LAMBDA * actor.layer(l)->node(n)->weight(i);

                double updated_weight = actor.layer(l)->node(n)->weight(i) + ALPHA * grad;
                actor.layer(l)->node(n)->set_weight(i, updated_weight);
            }
        }
    }
}

double optimize(Memory &memory, Net &critic, Net &target_critic, Net &actor, Net &target_actor) {
    std::vector<double> *state = memory.state();
    std::vector<double> *action = memory.action();

    std::vector<double> state_action;
    state_action.insert(state_action.end(), action->begin(), action->end());
    state_action.insert(state_action.end(), state->begin(), state->end());

    std::vector<double> q = critic.forward(state_action, false);
    double actor_loss = q[0];

    std::vector<double> *next_state = memory.next_state();
    std::vector<double> next_state_action = target_actor.forward(*next_state, false);
    next_state_action.insert(next_state_action.end(), next_state->begin(), next_state->end());

    std::vector<double> future = target_critic.forward(next_state_action, false);
    double optimal = memory.reward() + GAMMA * future[0];

    unsigned int num_of_tickers = state->size() / (OBS/INT);
    std::vector<double> agrad(num_of_tickers, 0.00);
    std::vector<bool> flag(num_of_tickers, false);

    std::thread critic_optimizer(optimize_critic, std::ref(critic), std::ref(state_action),
                                 optimal, q[0], num_of_tickers, std::ref(agrad), std::ref(flag));

    std::thread actor_optimizer(optimize_actor, std::ref(actor), std::ref(*state),
                                std::ref(*action), std::ref(agrad), std::ref(flag));

    critic_optimizer.join();
    actor_optimizer.join();

    std::vector<double>().swap(state_action);
    std::vector<double>().swap(q);
    std::vector<double>().swap(next_state_action);
    std::vector<double>().swap(future);
    std::vector<double>().swap(agrad);
    std::vector<bool>().swap(flag);

    return actor_loss;
}