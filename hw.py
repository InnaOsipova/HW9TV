# Задача 1 Даны значения величины заработной платы заемщиков банка (zp) и значения их
# поведенческого кредитного скоринга (ks): zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]. Используя математические
# операции, посчитать коэффициенты линейной регрессии, приняв за X заработную плату
# (то есть, zp - признак), а за y - значения скорингового балла (то есть, ks - целевая
# переменная). Произвести расчет как с использованием intercept, так и без

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


x = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110]).reshape((-1, 1))
y = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

model = LinearRegression()

regres = model.fit(x,y)
b0 = regres.intercept_
b1 = regres.coef_

y_pred = model.predict(x)

print(y_pred)

plt.scatter(x,y)
plt.plot(x, b0 + b1*x )
plt.show
plt.pause(30)

# Задача 2 Посчитать коэффициент линейной регрессии при заработной плате (zp), используя
# градиентный спуск (без intercept).

x = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110]).reshape((-1, 1))
y = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])
def mse_ (b1, y = y, x = x, n = len(x)):
    return np.sum((b1 * x - y)**2)/n

alfa = 0.000001
n = len(x)
b1 = 0.1
for i in range(1, 3000):
    b1 -= alfa * (2/n) * np.sum((b1 * x - y)*x)
    if i % 500 == 0 :
        print(f' Iteration = {i}, B1 = {b1}, MSE = {mse_(b1)}')
