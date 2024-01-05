#include <cstdlib>
#include <vector>
#include <string>
#include <fstream>

#include "../lib/data.hpp"

std::vector<std::vector<double>> read_csv(std::string path) {
    std::ifstream file(path);
    std::vector<std::vector<double>> dat;

    if(!file.is_open()) return dat;

    std::string line;
    file >> line;

    unsigned int columns = 1;
    for(char &ch: line)
        columns += (ch == ',');

    dat.resize(columns, std::vector<double>());
    
    double val = RAND_MAX;
    while(file >> line) {
        for(unsigned int col = 0; col < columns; col++) {
            val = std::stod(line.substr(0, line.find(",")));
            dat[col].push_back(val);
            line = line.substr(line.find(",") + 1);
        }
    }

    file.close();
    return dat;
}