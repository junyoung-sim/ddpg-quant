#include <cstdlib>
#include <vector>
#include <string>
#include <random>

#include "../lib/net.hpp"

double relu(double x) { return std::max(0.00, x); }
double drelu(double x) { return x < 0.00 ? 0.00 : 1.00; }

double Node::bias() { return b; }
double Node::sum() { return s; }
double Node::act() { return z; }
double Node::err() { return e; }
double Node::weight(unsigned int index) { return w[index]; }

void Node::init() { s = z = e = 0.00; }
void Node::set_bias(double val) { b = val; }
void Node::set_sum(double val) { s = val; }
void Node::set_act(double val) { z = val; }
void Node::add_err(double val) { e += val; }
void Node::set_weight(unsigned int index, double val) { w[index] = val; }

unsigned int Layer::in_features() { return in; }
unsigned int Layer::out_features() { return out; }
Node *Layer::node(unsigned int index) { return &n[index]; }

void Net::add_layer(unsigned int in, unsigned int out) { layers.push_back(Layer(in, out)); }
void Net::init(std::default_random_engine &seed, bool sm) {
    softmax = sm;
    std::normal_distribution<double> std_normal(0.00, 1.00);
    for(unsigned int l = 0; l < layers.size(); l++) {
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            for(unsigned int i = 0; i < layers[l].in_features(); i++)
                layers[l].node(n)->set_weight(i, std_normal(seed) * sqrt(2.00 / layers[l].in_features()));
        }
    }
}

std::vector<double> Net::forward(std::vector<double> &x) {
    std::vector<double> yhat; double expsum = 0.00;
    for(unsigned int l = 0; l < layers.size(); l++) {
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            double dot = 0.00;
            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                if(l == 0)
                    dot += x[i] * layers[l].node(n)->weight(i);
                else
                    dot += layers[l-1].node(i)->act() * layers[l].node(n)->weight(i);
            }

            layers[l].node(n)->init();
            layers[l].node(n)->set_sum(dot + layers[l].node(n)->bias());

            if(l == layers.size() - 1) {
                if(softmax) expsum += exp(layers[l].node(n)->sum());
                else yhat.push_back(layers[l].node(n)->sum());
                continue;
            }

            layers[l].node(n)->set_act(relu(layers[l].node(n)->sum()));
        }
    }

    if(softmax) {
        for(unsigned int n = 0; n < layers.back().out_features(); n++) {
            double norm = exp(layers.back().node(n)->sum()) / expsum;
            yhat.push_back(norm);
        }
    }

    return yhat;
}