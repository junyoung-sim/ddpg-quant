#include <cstdlib>
#include <iostream>
#include <random>

#include "../lib/param.hpp"
#include "../lib/data.hpp"
#include "../lib/gbm.hpp"
#include "../lib/net.hpp"

int main(int argc, char *argv[])
{
    std::default_random_engine seed;

    //std::system("./python/download.py SPY");
    std::vector<double> raw = read_csv("./data/merge.csv")[0];

    for(int i = raw.size() - 5; i < raw.size(); i++)
        std::cout << raw[i] << " ";
    std::cout << "\n";

    std::vector<double> v;
    vscore(raw, v, VOBS, VEXT, VITR, seed);

    for(int i = v.size() - 5; i < v.size(); i++)
        std::cout << v[i] << " ";
    std::cout << "\n";

    std::vector<double> x = {-1.0, 0.0, 0.3, 0.2, 0.8};

    Net qnet;
    qnet.add_layer(5, 5);
    qnet.add_layer(5, 5);
    qnet.add_layer(5, 3);
    qnet.init(seed, false);

    std::vector<double> yhat = qnet.forward(x);
    for(double &x: yhat)
        std::cout << x << " ";
    std::cout << "\n";

    Net policy;
    policy.add_layer(5, 5);
    policy.add_layer(5, 5);
    policy.add_layer(5, 3);
    policy.init(seed, true);

    std::vector<double>().swap(yhat);
    yhat = policy.forward(x);
    for(double &x: yhat)
        std::cout << x << " ";
    std::cout << "\n";

    return 0;
}