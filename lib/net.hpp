#ifndef __NET_HPP_
#define __NET_HPP_

#include <cstdlib>
#include <vector>
#include <random>

class Node
{
private:
    double b;
    double s;
    double z;
    double e;
    std::vector<double> w;
public:
    Node() {}
    Node(unsigned int in) {
        b = s = z = e = 0.00;
        w.resize(in, 0.00);
    }

    double bias();
    double sum();
    double act();
    double err();
    double weight(unsigned int index);

    void init();
    void set_bias(double val);
    void set_sum(double val);
    void set_act(double val);
    void add_err(double val);
    void set_weight(unsigned int index, double val);
};

class Layer
{
private:
    std::vector<Node> n;
    unsigned int in;
    unsigned int out;
public:
    Layer () {}
    Layer(unsigned int i, unsigned int o) {
        in = i; out = o;
        n.resize(out, Node(in));
    }

    unsigned int in_features();
    unsigned int out_features();

    Node *node(unsigned int index);
};

class Net
{
private:
    bool softmax;
    std::vector<Layer> layers;
public:
    Net() {}

    void add_layer(unsigned int in, unsigned int out);
    void init(std::default_random_engine &seed, bool sm);

    std::vector<double> forward(std::vector<double> &x);
    void backward(std::vector<double> &x, std::vector<double> &y, double alpha, double lambda);
};

double relu(double x);
double drelu(double x);

#endif