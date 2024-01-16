#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>

#include "../lib/gbm.hpp"

double mean(std::vector<double> &dat) {
    double sum = 0.00;
    for(double &x: dat)
        sum += x;
    return sum / dat.size();
}

double stdev(std::vector<double> &dat) {
    double mu = mean(dat);
    double rss = 0.00;
    for(double &x: dat)
        rss += pow(x - mu, 2);
    return sqrt(rss / dat.size());
}

void normal(std::vector<std::vector<double>> &dat, std::default_random_engine &seed) {
    std::normal_distribution<double> std_normal(0.0, 1.0);
    for(int i = 0; i < dat.size(); i++)
        for(int j = 0; j < dat[i].size(); j++) dat[i][j] = std_normal(seed);
}

void cumsum(std::vector<std::vector<double>> &dat) {
    for(int i = 0; i < dat.size(); i++)
        for(int j = 0; j < dat[i].size(); j++) dat[i][j] += dat[i][j-1];
}

std::vector<double> returns(std::vector<double> &raw) {
    std::vector<double> r;
    for(int i = 1; i < raw.size(); i++)
        r.push_back((raw[i] - raw[i-1]) / raw[i-1]);
    return r;
}

void vscore(std::vector<double> &raw, std::vector<double> &v,
            unsigned int obs, unsigned int ext, unsigned int itr, std::default_random_engine &seed) {
    for(int t = obs-1; t < raw.size(); t++) {
        std::vector<double> window = {raw.begin() + t+1-obs, raw.begin() + t+1};
        std::vector<double> r = returns(window);

        double s0 = window.back();
        double mu = mean(r);
        double sigma = stdev(r);
        double drift = mu + 0.5 * pow(sigma, 2);

        std::vector<std::vector<double>> path(itr, std::vector<double>(ext));
        normal(path, seed); cumsum(path);

        unsigned int sum = 0;
        for(int i = 0; i < itr; i++) {
            for(int j = 0; j < ext; j++) {
                path[i][j] *= sigma;
                path[i][j] += drift * (j+1);
                path[i][j] = s0 * exp(path[i][j]);
                sum += (path[i][j] > s0);
            }
        }
        v.push_back((double)sum / (itr * ext));
    }
}