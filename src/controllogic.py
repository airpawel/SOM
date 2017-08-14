import math
import random
import numpy as np
import matplotlib.pyplot as plt
from PyQt4 import QtCore
from PyQt4.Qt import QObject
from src.learningasset import LearningAsset
from src.datadeliver import DataDeliver
from src.neuron import Neuron
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()


class Control(QObject):

    newValue = QtCore.pyqtSignal()
    gammaChanged = QtCore.pyqtSignal(str, name='gammaChanged')
    sigmaChanged = QtCore.pyqtSignal(str, name='sigmaChanged')
    somMapChanged = QtCore.pyqtSignal(list, int, name='somMapCHanged')
    # setSomMapPixEdgeNum = QtCore.pyqtSignal(int, name='setSomMapPixEdgeNum')

    def __init__(self, dataset=None, parent=None):
        super(QObject, self).__init__(parent)
        self.dataset = dataset
        self.alfa = 1000
        self.sigma_0 = 16
        self.gamma_0 = 1
        self.network_size = 16
        self.neurons_num = int(math.pow(self.network_size, 2))
        self.winner_coordinates = None
        self.step_result = None
        self.delta = None
        self.gamma = None
        self.deltas = []

        self.neurons_w_num = dataset.params_num
        self.__initialize_input(dataset)
        self.__initalize_output()

        self.mcolor = np.array([[1, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 0], [0, 1, 1], [1, 1, 1]])
        if self.neurons_w_num < 7:
            self.colors = self.mcolor[:self.neurons_w_num]
        else:
            self.colors = self.mcolor[:]

        # this one is tricky because is only used to mix population ids and
        self.population_of_ids = [i for i in range(0, len(self.input))]
        # will be controled by
        self.input_random_order = True

        self.data_deliver = DataDeliver()

    def __str__(self):
        text = 'Network map:\n'
        for i in range(0, self.network_size):
            for j in range(0, self.network_size):
                text += '({},{}): {} | '.format(i+1, j+1, self.output[i][j])
            text += '\n'

        return text

    def get_params_bandwidth(self):
        ''' get bandwidth of input vectors to adjust generation of SOM map (see Neuron) '''
        p = np.array([])
        for item in self.input:
            p = np.append(p, item.params)
        p = p.reshape((len(self.input), self.neurons_w_num))
        l_bound = p.min(axis=0)
        u_bound = p.max(axis=0)
        return u_bound - l_bound

    def __initialize_input(self, dataset):
        """ draws only elements from dataset
        :param dataset: instance of LearningAsset class holding data records
        """
        self.input = []
        for i in range(0, dataset.elements_num):
            self.input.append(dataset.elements[i])

    def __initalize_output(self):
        """ builds SOM map of neurons """
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
        # reshape output SOM map from 2-dim to 1-dim in the end its shape will be restored
        self.output = self.output.reshape((1, self.neurons_num))
        for neuron in self.output[0]:
            norms.append(self.euclidean(neuron.weights, v))
        # restoration of SOM map shape
        self.output = self.output.reshape((self.network_size, self.network_size))
        return np.array(norms).reshape((self.network_size, self.network_size))

    def dist2best(self, n, c):
        return np.sqrt(np.sum((n - c) ** 2))

    def i_steps_forward_handler(self, k, i):
        for p in range(k, k+i):
            self.one_step_learn(p)
        self.gammaChanged.emit(str(self.gamma))
        self.sigmaChanged.emit(str(self.sigma))
        self.somMapChanged.emit(self.step_result.tolist(), self.network_size)

    def som2show(self, i, j):
        w_max = np.max(self.output[i][j].weights)
        normalized_w = np.divide(self.output[i][j].weights, w_max)
        f1 = 40 / self.neurons_w_num
        k1 = np.dot(np.multiply(normalized_w, f1), self.colors)
        return k1

    def one_step_learn(self, k):
        k2 = np.array([])
        self.deltas = []
        if self.input_random_order:
            id = random.choice(self.population_of_ids)


        # those parameters change after every k iteration
        # at first step k == 0 so gamma_0 and sigma_0 are considered valid at first
        self.gamma = self.gamma_0 * np.exp(-k / self.alfa)
        self.sigma = self.sigma_0 * np.exp(-k / self.alfa)
        X = self.input[id]
        # print(self.input[id])

        # matix of euclidean norms between (neuron.weights, input) data for every neuron
        norms = self.get_norms(X.params)
        location_of_min = np.argmin(norms)

        # compute (i, j) coordinates of maximum value in norms matrix
        self.winner_coordinates = np.array([int(location_of_min / self.network_size), location_of_min % self.network_size])

        # step over neighbourhood and update
        for i in range(0, self.network_size):
            for j in range(0, self.network_size):
                neighbour = np.array([i, j])
                # distance to champion in pixels
                dn = self.dist2best(neighbour, self.winner_coordinates)
                # update neighbours weights
                if dn < self.sigma:
                    self.delta = np.exp((- dn ** 2) / (2 * (self.sigma ** 2)))
                    self.deltas.append(self.delta)
                    self.output[i][j].weights += self.delta * self.gamma * (X.params - self.output[i][j].weights)
                # this part is for visualization
                k1 = np.dot(np.multiply(self.output[i][j].weights[:3], 10), self.mcolor[:3])
                k2 = np.append(k2, k1)
        self.step_result = k2.reshape((self.network_size, self.network_size, 3))

    def one_step_learn_call_handler(self, index):
        self.one_step_learn(index)
        self.gammaChanged.emit(str(self.gamma))
        self.sigmaChanged.emit(str(self.sigma))
        self.somMapChanged.emit(self.step_result.tolist(), self.network_size)

    def change_alfa(self, v):
        self.alfa = v
        print('alfa: ', self.alfa)

    def change_gamma_zero(self, v):
        self.gamma_0 = v
        print('gamma 0: ', self.gamma_0)

    def change_sigma_zero(self, v):
        self.sigma_0 = v
        print('sigma 0: ', self.sigma_0)

    def change_network_size(self, v):
        self.network_size = v
        self.neurons_num = int(math.pow(self.network_size, 2))
        self.__initalize_output()
        print('network size: ', self.network_size, ' neurons: ', self.neurons_num)

    def change_output_data(self):
        pass


if __name__ == '__main__':
    # this does not work properly, was written only for tests
    data = LearningAsset()
    data.loadAsset('data/IrisDataAll.csv')
    cont = Control(dataset=data)

    for k in range(0, 500):
        cont.one_step_learn(k)
        # if np.amax(cont.output) > 255:
        # print('delta: ', cont.delta, 'gamma: ', cont.gamma, 'delta * gamma: ', cont.delta * cont.gamma)
        winner = lambda x,y: cont.output[x][y]
        print('WINNER: ', winner(*cont.winner_coordinates))

    img = cont.step_result
    print(type(img[0][0]))
    print(img.shape)
    print(np.amax(img))
    print(np.amin(img))
    img.astype('int32')
    plt.figure()
    plt.show()


