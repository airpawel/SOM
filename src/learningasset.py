import csv
from src.dataentity import DataEntity
import os


class LearningAsset(object):
    def __init__(self, d=[], h=[]):
        # order of parameters assignment is important here (check property for them)
        self.elements = d
        self.headers = h
        self.num = {}

    @property
    def elements_num(self):
        return len(self.elements)

    @property
    def params_num(self):
        # print('params_num')
        # it is enough to check if length of elements is > 0 because during
        # loading process only apropriate data is loaded
        if len(self.__elements) != 0:
            return self.__elements[0].params_num
        else: return None

    @property
    def headers(self):
        return self.__headers

    @headers.setter
    def headers(self, h):
        # check if data loaded list is not empty
        if self.params_num:
            # add labels for parameters if passed headers h does not contain all of them
            if len(h) < self.params_num:
                # generate labels for headers
                for i in range(len(h), self.params_num):
                    h.append('param-%d' % (i+1,))
                self.__headers = h
            # take only n first labels from h because there is less params than labels
            elif len(h) > self.params_num:
                print('greater')
                self.__headers = h[:self.params_num]
            else:
                self.__headers = h
        else:
            self.__headers = []

    @property
    def elements(self):
        return self.__elements

    @elements.setter
    def elements(self, d):
        data = []
        # d needs to be list with some elements and its elements need to be lists
        if (len(d) != 0):
            # if there is a list we assume it is a list of numbers
            if isinstance(d[0], list):
                # this is main loop that loads numeric data into DataEntity objects
                for i in range(0, len(d)):
                    try:
                        a = DataEntity(d[i])
                        data.append(a)
                    except Exception as e:
                        print(e)
                self.__elements = data
            elif isinstance(d[0], DataEntity):
                self.__elements = d
        else:
            self.__elements = []

    def __iter__(self):
        """ this function is an iterator, it is called implicitly at the begining
            of every loop to retreive the object of the class to which it belongs
            :return: object of the class
        """
        self.i = 0
        return self

    def __next__(self):
        """ this function is fired every time the loop ends one iteration,
            it makes it possible to iterate over object to get its elements,
            one by one, iteration by iteration
            :return: following elements of the object (iterable object!!)
        """
        if self.i >= len(self.elements):
            raise StopIteration
        self.i += 1
        return self.elements[self.i-1]

    @property
    def minimal_set_num(self):
        statistics = {}
        for e in self.elements:
            if e.name in statistics: statistics[e.name] += 1
            else:                    statistics[e.name] = 1
        num = [v for k, v in statistics.items()]
        return min(num) if len(num) != 0 else 0

    def __str__(self):
        """ it may be suprising here but in for loop we can use self.elements
            as well as self, since there was indexing implemented for this object
        """
        # reverse order of labels
        # text = '  | '.join(self.headers[::-1]) + "\n"
        text = '  | '.join(self.headers[:]) + "\n"
        for e in self.elements:
            text += e.__str__() + '\n'
        return text

    def loadAsset(self, filepath):
        h = []
        with open(filepath) as file:
            reader = csv.reader(file, delimiter=';')
            for i, row in reversed(list(enumerate(reader))):
                # print(i, row)
                if i != 0:
                    name = row.pop(len(row) - 1).strip()
                    params = []
                    while len(row) > 0:
                        params.append(float(row.pop(0)))
                    # print(params, name)
                    item = DataEntity(params, name)
                    self.elements.append(item)
                else:
                    h.extend(row[:-1])
                    # print(row[:-1])
            self.headers = h

