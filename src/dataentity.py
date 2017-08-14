import numpy as np


class DataEntity(object):
    num_of_entities = 0

    def __init__(self, p, *argv):
        self.id = DataEntity.num_of_entities
        DataEntity.num_of_entities += 1
        self.params = np.array(p)
        try:               self.name = argv[0] if argv[0] != '' else '-'
        except IndexError: self.name = '-'

    @property
    def params_num(self):
        return len(self.__params)

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, n):
        """ this property function only imposes some conditions to data class name
        """
        n = n.split()[0]
        if len(n) > 25: self.__name = n[:25]
        else:           self.__name = n

    @property
    def params(self):
        return self.__params

    @params.setter
    def params(self, p):
        """ very important is that DataEntity cannot exist without data otherwise
            throws exception when object is being created
        """
        if p.size == 0: raise Exception('Empty list of parameters. Object DataEntity not created!')
        self.__params = np.array(p)

    def __iter__(self):
        """ is called implicitly at start of every loop
        :return: returns objects reference
        """
        self.i = 0
        return self

    def __next__(self):
        """ used implicitly at the end of every iteration step
        :return: each following element of the DataEntity params list
        """
        if self.i >= len(self.params):
            raise StopIteration
        else:
            self.i += 1
            return self.params[self.i]

    def __del__(self):
        """ deletes object previously dealing with objects number which is
            decreased by one
        """
        DataEntity.num_of_entities -= 1
        del self

    def __str__(self):
        np.set_printoptions(formatter={'float': lambda v: '{: 0.3f}'.format(v)})
        return '{:<5} {:25} {}'.format(self.id, self.name, str(self.params)[1:-1])
