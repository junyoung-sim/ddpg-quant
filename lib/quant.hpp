#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>

#include "../lib/param.hpp"
#include "../lib/data.hpp"
#include "../lib/gbm.hpp"
#include "../lib/net.hpp"

class Memory
{
private:
    std::vector<double> s0;
    std::vector<double> a;
    std::vector<double> s1;
    double r;
public:
    Memory() {}
    Memory(std::vector<double> &current, std::vector<double> &action,
           std::vector<double> &next, double reward) {
        s0.swap(current);
        a.swap(action);
        s1.swap(next);
        r = reward;
    }
    ~Memory() {
        std::vector<double>().swap(s0);
        std::vector<double>().swap(a);
        std::vector<double>().swap(s1);
    }

    std::vector<double> *state() { return &s0; }
    std::vector<double> *action() { return &a; }
    std::vector<double> *next_state() { return &s1; }
    double reward() { return r; }
};

std::vector<double> sample_state(std::vector<std::vector<double>> &env, unsigned int t);

std::vector<double> epsilon_greedy(Net &actor, std::vector<double> &state, double eps);

void build(std::vector<std::string> &tickers, std::vector<std::vector<double>> &price,
           std::vector<std::vector<double>> &valuation, Net &actor, Net &critic, std::default_random_engine &seed);

void optimize(Memory &memory, Net &critic, Net &target_critic,
              Net &actor, Net &target_actor, double alpha, double lambda);


#endif