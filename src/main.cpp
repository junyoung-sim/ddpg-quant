#include <cstdlib>
#include <iostream>
#include <random>
#include <chrono>
#include <string>
#include <vector>
#include <thread>

#include "../lib/quant.hpp"

int main(int argc, char *argv[])
{
    std::string mode, name, cmd;
    std::string actor_path, critic_path;
    std::vector<std::string> tickers;
    std::vector<std::vector<double>> price;
    std::vector<std::vector<double>> valuation;
    std::vector<std::thread> threads;

    Net actor, target_actor;
    Net critic, target_critic;

    std::default_random_engine seed(std::chrono::system_clock::now().time_since_epoch().count());

    mode = argv[1];
    name = argv[2];
    actor_path = "./models/" + name + "-actor";
    critic_path = "./models/" + name + "-critic";

    cmd = "./python/download.py ";
    for(unsigned int i = 3; i < argc; i++) {
        tickers.push_back(argv[i]);
        cmd += tickers.back() + " ";
    }
    std::system(cmd.c_str());

    price = read_csv("./data/merge.csv");
    valuation.resize(tickers.size(), std::vector<double>());

    for(unsigned int i = 0; i < tickers.size(); i++) {
        std::thread th(vscore, std::ref(price[i]), std::ref(valuation[i]), VOBS, VEXT, VITR, std::ref(seed));
        threads.push_back(std::move(th));
    }
    for(std::thread &th: threads) th.join();
    for(std::vector<double> &p: price) p.erase(p.begin(), p.begin() + VOBS-1);

    actor.add_layer(tickers.size() * OBS, tickers.size() * OBS);
    actor.add_layer(tickers.size() * OBS, tickers.size() * OBS);
    actor.add_layer(tickers.size() * OBS, tickers.size() * OBS);
    actor.add_layer(tickers.size() * OBS, tickers.size()); actor.use_softmax();
    actor.init(seed);
    actor.load(actor_path);

    critic.add_layer(tickers.size() * (OBS+1), tickers.size() * (OBS+1));
    critic.add_layer(tickers.size() * (OBS+1), tickers.size() * (OBS+1));
    critic.add_layer(tickers.size() * (OBS+1), tickers.size() * (OBS+1));
    critic.add_layer(tickers.size() * (OBS+1), 1);
    critic.init(seed);
    critic.load(critic_path);

    copy(actor, target_actor);
    copy(critic, target_critic);

    actor.save(actor_path);
    critic.save(critic_path);

    return 0;
}