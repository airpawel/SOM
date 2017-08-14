import os
import time
import json
import jsonpickle
import random
import numpy as np
import matplotlib.pyplot as plt
from src.dataentity import DataEntity
from src.learningasset import LearningAsset
from src.neuron import Neuron
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()


class SomNetwork(object):

    def __init__(self, dataset):
        self.neurons_w_num = dataset.params_num
        self.__network_size(dataset.elements_num)
        self.__neurons_num(dataset.elements_num)
        self.__initialize_input(dataset)
        self.__initalize_output()
        self.winner_coordinates = None
        self.result = None
        # adaptation parameters
        self.sigma_0 = self.network_size - 4
        self.gamma_0 = 1
        self.alfa = 4000
        self.mcolor = np.array([[255, 255, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
        self.population_of_ids = [i for i in range(0, len(self.input))]

    def test_fun(self):
        pass

    def __str__(self):
        text = 'Network map:\n'
        for i in range(0, self.network_size):
            for j in range(0, self.network_size):
                text += '({},{}): {} | '.format(i+1, j+1, self.output[i][j])
            text += '\n'

        return text

    def get_params_bandwidth(self):
        p = np.array([])
        for item in self.input:
            p = np.append(p, item.params)
        p = p.reshape((len(self.input), self.neurons_w_num))
        l_bound = p.min(axis=0)
        u_bound = p.max(axis=0)
        print(u_bound - l_bound)
        return u_bound - l_bound

    def __network_size(self, n):
        """ returns dimension (N) for the network map (N x N)
        :param n: number of input vectors - dataset records
        """
        # self.network_size = int(np.ceil(np.sqrt(5 * np.sqrt(n))))
        self.network_size = 12

    def __neurons_num(self, n):
        """ number of total neurons in the SOM map
        :param n: number of input vectors - dataset records
        """
        d = np.ceil(np.sqrt(5 * np.sqrt(n)))
        # self.neurons_num = int(d * d)
        self.neurons_num = 144

    def __initialize_input(self, dataset):
        """ draws only elements from dataset
        :param dataset: instance of LearningAsset class holding data records
        """
        self.input = []
        for i in range(0, dataset.elements_num):
            self.input.append(dataset.elements[i])

    def __initalize_output(self, fromfile=True):
        """ builds SOM map of neurons """
        # self.read_network_map()
        # self.__show_output(result)
        # self.test_fun()
        # self.read_network_map()
        # print('initialize output')

        self.output = np.array([])
        # numpy array of max_i - min_i for every i-parameter
        bandwidth = self.get_params_bandwidth()
        for i in range(0, self.neurons_num):
            self.output = np.append(self.output, Neuron(self.neurons_w_num, bandwidth))
        self.output = np.reshape(self.output, (self.network_size, self.network_size))

    # noinspection PyTypeChecker
    def euclidean(self, X, Y):
        """ Euclidean norm for two numpy.ndarrays X and Y """
        return np.sqrt(np.sum((X - Y) ** 2))

    def get_norms(self, v):
        norms = []
        self.output = self.output.reshape((1, self.neurons_num))
        for neuron in self.output[0]:
            norms.append(self.euclidean(neuron.weights, v))
        self.output = self.output.reshape((self.network_size, self.network_size))
        return np.array(norms).reshape((self.network_size, self.network_size))

    def distance_to_champion(self, n, c, metric):
        if metric == 'taxicab':
            return np.sum(np.abs(n - c))
        else:
            return np.sqrt(np.sum((n - c) ** 2))

    def learn(self):
        # # adaptation parameters
        # sigma_0 = self.network_size - 4
        # gamma_0 = 1
        # alfa = 4000
        # mcolor = np.array([[255, 255, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]])
        #
        # population_of_ids = [i for i in range(0, len(self.input))]
        # rand_ids = random.sample([i for i in range(0, len(self.input))], len(self.input))
        for k in range(0, 10):
            k3 = self.one_step_learn(k, self.sigma_0, self.gamma_0, self.alfa, self.mcolor, self.population_of_ids)
            k4 = k3.reshape((network.network_size, network.network_size, 3))
            # time.sleep(0.2)
        return k4

    def one_step_learn(self, k, sigma_0, gamma_0, alfa, mcolor, population_of_ids):
        k3 = np.array([])
        id = random.choice(population_of_ids)
        # those parameters change after every k iteration
        # at first step k == 0 so gamma_0 and sigma_0 are considered valid at first
        gamma = gamma_0 * np.exp(-k / alfa)
        sigma = sigma_0 * np.exp(-k / alfa)
        d = self.input[id]
        # print(self.input[id])

        # matix of euclidean norms between neuron weights and input data for every neuron
        norms = self.get_norms(d.params)
        location_of_min = np.argmin(norms)

        # compute (i, j) coordinates of maximum value in norms matrix
        self.winner_coordinates = np.array([int(location_of_min / 8), location_of_min % 8])
        # print(k, self.winner_coordinates)
        print('inside: ', k)
        # step over neighbourhood and update
        for i in range(0, self.network_size):
            for j in range(0, self.network_size):
                neighbour = np.array([i, j])
                dn = self.distance_to_champion(neighbour, self.winner_coordinates, 'euclidean')
                if dn < sigma:
                    # update neighbours weights
                    delta = np.exp((- dn ** 2) / (2 * (sigma ** 2)))
                    self.output[i][j].weights += delta * gamma * (d.params - self.output[i][j].weights)
                    pass
                k1 = np.dot(self.output[i][j].weights, mcolor)
                # k2 = np.sum(k1, axis=1)
                k3 = np.append(k3, k1)
        k4 = k3.reshape((self.network_size, self.network_size, 3))
        self.result = k4
        return k4


if __name__ == '__main__':
    data = LearningAsset()
    data.loadAsset('../data/IrisDataAll.csv')
    # print(data)
    network = SomNetwork(data)
    # print(network.output)
    # network.save_network_map()
    # print('\nSOM map data was dumped into data/map.json.\n')
    # network.read_network_map()
    # print('\nSOM map data was read from data/map.json.\n')

    # im = network.learn()
    # plt.figure()
    # plt.imshow(im)
    # plt.show()
    # print(network)

    # mcolor = np.array([[255,255,0],[255,0,0],[0,255,0],[0,0,255]])
    # k3 = np.array([])
    # for i in range(0, network.network_size):
    #     for j in range(0, network.network_size):
    #         k1 = np.dot(network.output[i][j].weights, mcolor)
    #         # k2 = np.sum(k1, axis=1)
    #         k3 = np.append(k3, k1)
    # k4 = k3.reshape((network.network_size, network.network_size, 3))
    # # plt.imshow(k4, interpolation='nearest')
    # plt.imshow(k4)
    # plt.show()

    # print(network)
    # norms = network.get_norms(network.input[0].params)
    # print(norms)
    # m = np.max(norms)
    # print(m)
    # print(network.output.shape)
    # idx = np.argmax(norms)
    # network.winner_coordinates = [int(idx/8), idx%8]
    # print(idx)
    # print('x - ',idx%8)
    # print('y - ',int(idx/8))



