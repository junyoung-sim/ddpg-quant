#ifndef __QUANT_HPP_
#define __QUANT_HPP_

#include <cstdlib>
#include <vector>

#include "../lib/param.hpp"
#include "../lib/data.hpp"
#include "../lib/gbm.hpp"
#include "../lib/net.hpp"

std::vector<double> sample_state(std::vector<std::vector<double>> &env, unsigned int t);

#endif