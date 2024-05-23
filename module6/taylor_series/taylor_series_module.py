import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

def my_taylor_series(func, x0, n_terms):
    """
    Розклад у ряд Тейлора без використання вбудованих функцій.
    :param func: Функція, яку будемо розкладати.
    :param x0: Точка розкладання.
    :param n_terms: Кількість членів ряду Тейлора.
    :return: Ряд Тейлора
    """
    taylor_series = 0
    
    for n in range(n_terms):
    # Обчислення n-го члена ряду Тейлора
        term = func.diff(x, n).subs(x, x0) / sp.factorial(n) * (x - x0)**n
        taylor_series += term
    return taylor_series

# Визначення символьної змінної та функції
x = sp.symbols('x')
func = sp.sin(x) + 6*sp.cos(x)+- x**2

# Точка розкладання та кількість членів ряду Тейлора
x0 = 0
n_terms = 15

# Розклад у ряд Тейлора
taylor_series = my_taylor_series(func, x0, n_terms)

# Компіляція функцій для NumPy
func_np = sp.lambdify(x, func, 'numpy')
taylor_np = sp.lambdify(x, taylor_series, 'numpy')

# Генерація значень для графіків
x_vals = np.linspace(-2 * np.pi, 2 * np.pi, 100)
y_vals_func = func_np(x_vals)
y_vals_taylor = taylor_np(x_vals)

# Побудова графіка
plt.plot(x_vals, y_vals_func, label='Функція')
plt.plot(x_vals, y_vals_taylor, label=f'Ряд Тейлора (до {n_terms}-го члена)')
plt.title(f'Розклад у ряд Тейлора для sin(x) в точці x={x0}')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
