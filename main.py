import Predictor as pr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator

# Данные длины
inputs = [12, 2, 10, 3, 11, 4, 12, 5, 8, 2, 9, 2, 10, 3]
# inputs = [2, 9, 3, 10, 4, 11, 6, 12, 3, 11, 4, 12, 5, 12]


# Данные ширины
targets = [2, 9, 3, 10, 4, 11, 6, 12, 3, 11, 4, 12, 5, 12]
# targets = [12, 2, 10, 3, 11, 4, 12, 5, 8, 2, 9, 2, 10, 3]

weight = 0.05
learningRate = 0.1

p = pr.LinearClassifier(weight, learningRate)

print("----------before trained----------")

g = []
k = 0 # счетчик
# Данные до обучения
for i, t, in zip(inputs, targets):
    k = k + 1 # счетчик
    # Если при линейной класификации массива через его длину мы 
    # получаем значение на линии меньше его истинного значения ширины 
    # при этом же значении длины без линейной класификации, то это Массив 1,
    # если меньше, то Массив 2
    if p.query(i) <= t:
        print(f"Приклад {k}:", " Массив 1 -- ", round(p.query(i), 2))
    else:
        print(f"Приклад {k}:", " Массив 2 -- ", round(p.query(i), 2))
    g.append(p.query(i))

for i, t, in zip(inputs, targets):
    p.train(i, t)

print("----------after trained----------")
k = 0

j = []
for i, t in zip(inputs, targets):
    k = k + 1
    # Истинное значение ширины больше чем длины после линейной класификации, 
    # то это Массив 1 
    if p.query(i) <= t:
        print(f"Приклад {k}:", " Массив 1 -- ", round(p.query(i), 4))
    else:
        print(f"Приклад {k}:", " Массив 2 -- ", round(p.query(i), 4))
    j.append(p.query(i))


fig = plt.figure()
gs = GridSpec(ncols=2, nrows=1, figure=fig)

ax1 = plt.subplot(gs[0, 0])
plt.title('До обучения')
ax1.plot(np.array(inputs), np.array(g))
ax1.scatter(inputs, targets)
plt.text(4, 11, "Массив 1", color="blue")
plt.text(10.6, 3, "Массив 2", color="blue")
plt.xlabel('Длина')
plt.ylabel('Ширина')
ax1.xaxis.set_major_locator(FixedLocator(inputs))
ax1.yaxis.set_major_locator(FixedLocator(targets))
plt.grid()

ax2 = plt.subplot(gs[0, 1:])
plt.title('После обучения')
ax2.plot(np.array(inputs), np.array(j))
ax2.scatter(inputs, targets)
plt.text(4, 11, "Массив 1", color="blue")
plt.text(10.6, 3, "Массив 2", color="blue")
plt.xlabel('Длина')
plt.ylabel('Ширина')
ax2.xaxis.set_major_locator(FixedLocator(inputs))
ax2.yaxis.set_major_locator(FixedLocator(targets + [13, 15, 17, 19, 21]))

plt.grid()
plt.show() 
