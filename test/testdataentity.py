from src.dataentity import DataEntity

# data parameters to load some are left empty to check if
# loading works properly
par =  [[],
        [],
        [1.1, 2.2222, 3.33, 4.444, 2],
        [1.1, 2.2222, 3.33, 4.444, 3],
        [1.1, 2.2222, 3.33, 4.444, 4],
        [1.1, 2.2222, 3.33, 4.444, 5]]

names = ['stolik maly', '', 'szafa      czarna', 'szafadozabudowy', '', 'jama smoka']

# list that collects all DataEntity objects created using par list
l = []

# loop for creating DataEntity objects and appending them to the list
# if there are problems with data exception is handled
for i in range(0, 6):
    try:
        a = DataEntity(par[i],names[i])
        l.append(a)
        print(a)
    except Exception as e:
        print(e)
