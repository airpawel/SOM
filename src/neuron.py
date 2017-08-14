import json
import jsonpickle
import numpy as np
from random import uniform
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

class Neuron(object):

    def __init__(self, n, b, data=[]):
        '''
        :param n: neuron length
        :param b: vector of bandwidth for every neuron weight more info in network (get_params_bandwidth)
        :param data: user data to initialize neuron with, when passed weights are not random generated
        '''
        self.weights = np.array(self.__produce_base_neuron_weights(n, b, data))
        self.winner = False

    # def __reduce__(self):
    #     ''' this function is used when pickle is used for dumps an loads '''
    #     return (self.__class__, (self.weights, self.winner,),)

    def __str__(self):
        np.set_printoptions(formatter={'float': lambda v: '{: 0.5f}'.format(v)})
        return '{}'.format(str(self.weights)[1:-1])

    def neuron2rgb(self, tmatrix):
        self.weights = np.dot(self.weights, tmatrix)

    def __mul__(self, p):
        return np.multiply(self.weights, p)

    def __produce_base_neuron_weights(self, n, bandwidth, data):
        if data:
            return data
        w = [uniform(bandwidth[i] / 100, bandwidth[i] / 10) for i in range(0, n)]
        return w


if __name__ == '__main__':
    # data preparation
    neurons = []
    n2 = None
    for i in range(144):
        neurons.append(Neuron(4, [1 for i in range(4)]))
    ndarr_neurons = np.array(neurons).reshape((12, 12,))

    # json encoding and json decoding
    jsonpickle.set_encoder_options('json', indent=4)
    json_str = jsonpickle.encode(ndarr_neurons)
    with open('neuron.json', 'w', encoding='utf-8') as fin:
        fin.write(json_str)
    with open('neuron.json', 'r', encoding='utf-8') as fout:
        file = fout.read()
        n2 = jsonpickle.decode(file)
    print('n1: ', ndarr_neurons[0][0])
    print('n2: ', n2[0][0])
    print(n2[0][0]*10)
    print(type(n2[0][0]))
    print(n2[0][0].neuron2rgb(np.array([[1,0,0],[0,1,0],[0,0,1],[0,0,0]])))
    print(n2[0][0])




