#ifndef __GBM_HPP_
#define __GBM_HPP_

#include <cstdlib>
#include <vector>
#include <random>

double mean(std::vector<double> &dat);
double stdev(std::vector<double> &dat);
void standardize(std::vector<double> &dat);

void normal(std::vector<std::vector<double>> &dat, std::default_random_engine &seed);
void cumsum(std::vector<std::vector<double>> &dat);

std::vector<double> returns(std::vector<double> &raw);

void vscore(std::vector<double> &raw, std::vector<double> &v,
            unsigned int obs, unsigned int ext, unsigned int itr, std::default_random_engine &seed);

#endif