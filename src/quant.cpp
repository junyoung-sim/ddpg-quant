#include <cstdlib>
#include <vector>
#include <string>

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
    return actor.forward(state, explore < eps);
}

