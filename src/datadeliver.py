import jsonpickle
import numpy as np
import os
import time
import random
from src.learningasset import LearningAsset
from src.network import SomNetwork
from src.neuron import Neuron
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

class DataDeliver(object):
    def __init__(self):
        self.data_ready = False

    def data_generation(self, x, y):
        neurons = []
        n2 = None
        for i in range(x * y):
            neurons.append(Neuron(4, [1 for i in range(4)]))
        ndarr_neurons = np.array(neurons).reshape((x, y,))
        return ndarr_neurons

    def save_network_map(self, data):
        # pickling
        jsonpickle.set_encoder_options('json', indent=4)
        json_str = jsonpickle.encode(data)
        with open(os.path.abspath('map.json'), mode='w', encoding='utf-8') as fin:
            fin.write(json_str)

    def read_network_map(self):
        obj = None
        file = None
        with open(os.path.abspath('map.json'), mode='r', encoding='utf-8') as fout:
            file = fout.read()
        obj = jsonpickle.decode(file)

import timeit
if __name__ == '__main__':
    dd = DataDeliver()
    data = LearningAsset()
    data.loadAsset('../data/IrisDataAll.csv')
    network = SomNetwork(data)
    dd.save_network_map(network.output)
    dd.read_network_map()


