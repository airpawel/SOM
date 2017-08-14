from src.learningasset import LearningAsset
import os

# data parameters to load some are left empty to check if
# loading works properly
par = [[1.1, 2.2222, 3.33, 4.444, 0],
       [1.1, 2.2222, 3.33, 4.444, 1],
       [1.1, 2.2222, 3.33, 4.444, 2],
       [1.1, 2.2222, 3.33, 4.444, 3],
       [1.1, 2.2222, 3.33, 4.444, 4],
       [1.1, 2.2222, 3.33, 4.444, 5]]

# names = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
names = ['p1', 'p2', 'p3']

par1 = []
# data = LearningAsset(par, names)
data = LearningAsset()
data.loadAsset('../data/IrisDataAll.csv')
print(data)

print('number of data elements')
print(data.elements_num)
