from src.prop import Prop

par =  [[],
        [],
        [1.1, 2.2222, 3.33, 4.444, 2],
        [1.1, 2.2222, 3.33, 4.444, 3],
        [1.1, 2.2222, 3.33, 4.444, 4],
        [1.1, 2.2222, 3.33, 4.444, 5]]
l = []
for i in range(0, 6):
    try:
        a = Prop(par[i])
        l.append(a)
        print(a)
    except Exception as e:
        print(e)


