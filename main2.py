import Predictor as pr

# Данные ширины
inputs = [2, 9, 3, 10, 4, 11, 6, 12, 3, 11, 4, 12, 5, 12]

# Данные длины
targets = [12, 2, 10, 3, 11, 4, 12, 5, 8, 2, 9, 2, 10, 3]


weight = 0.05
learningRate = 0.1

p = pr.LinearClassifier(weight, learningRate)

print("----------before trained----------")

g = []
k = 0 # счетчик
# Данные до обучения
for i, t, in zip(inputs, targets):
    k = k + 1 # счетчик
    if p.query(i) <= t:
        print(f"Приклад {k}:", " Массив 2 -- ", round(p.query(i), 2))
    else:
        print(f"Приклад {k}:", " Массив 1 -- ", round(p.query(i), 2))
    g.append(p.query(i))

for i, t, in zip(inputs, targets):
    p.train(i, t)

print("----------after trained----------")
k = 0

j = []
for i, t in zip(inputs, targets):
    k = k + 1
    if p.query(i) >= t:
        print(f"Приклад {k}:", " Массив 1 -- ", round(p.query(i), 4))
    else:
        print(f"Приклад {k}:", " Массив 2 -- ", round(p.query(i), 4))
    j.append(p.query(i))
    