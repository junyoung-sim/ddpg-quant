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
};

double relu(double x);
double drelu(double x);

/*
void Net::backward(std::vector<double> &x, std::vector<double> &y, double alpha, double lambda) {
    unsigned int targ;
    if(softmax) targ = std::max_element(y.begin(), y.end()) - y.begin();

    std::vector<double> yhat = forward(x);

    for(int l = layers.size() - 1; l >= 0; l--) {
        double part = 0.00, grad = 0.00;
        for(unsigned int n = 0; n < layers[l].out_features(); n++) {
            if(l == layers.size() - 1 && softmax) part = yhat[n] - (n == targ);
            else if(l == layers.size() - 1 && !softmax) part = -2.00 * (y[n] - yhat[n]);
            else part = layers[l].node(n)->err() * drelu(layers[l].node(n)->sum());
            
            double updated_bias = layers[l].node(n)->bias() - alpha * part;
            layers[l].node(n)->set_bias(updated_bias);

            for(unsigned int i = 0; i < layers[l].in_features(); i++) {
                if(l == 0) grad = part * x[i];
                else {
                    grad = part * layers[l-1].node(i)->act();
                    layers[l-1].node(i)->add_err(part * layers[l].node(n)->weight(i));
                }

                grad += lambda * layers[l].node(n)->weight(i);

                double updated_weight = layers[l].node(n)->weight(i) - alpha * grad;
                layers[l].node(n)->set_weight(i, updated_weight);
            }
        }
    }
}
*/

#endif