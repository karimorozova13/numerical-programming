import sympy as sp
import numpy as np

"""Розрахуємо Якобіан для системи функцій

f1(x,y,z)=x^2+y^2 

f2(x,y,z)=xyz 

f3(x,y,z)=x^3+z^3 

в точці  (1,2,4)
"""

def jacobian_matrix(funcs, vars, point):
    m = len(funcs)  # Кількість функцій у системі
    n = len(vars)   # Кількість змінних

    # Створюємо матрицю Якобіана як символьну матрицю
    jacobian = sp.Matrix(m, n, lambda i, j: sp.diff(funcs[i], vars[j]))

    # Підставляємо значення змінних у матрицю Якобіана для заданої точки
    jacobian_evaluated = jacobian.subs({var: val for var, val in zip(vars, point)})

    # Перетворюємо матрицю SymPy у масив NumPy
    jacobian_numerical = np.array(jacobian_evaluated.tolist(), dtype=float)

    return jacobian, jacobian_numerical


# Створюємо символьні змінні
x, y, z = sp.symbols('x y z')

# Задаємо систему функцій
f1 = x**2 + y**2
f2 = x*y*z
f3 = x**3+z**3

# Список функцій у системі
funcs = [f1, f2, f3]

# Список змінних
vars = [x, y, z]

# Задаємо точку, в якій потрібно обчислити матрицю Якобіана
point = (1, 2, 4)

# Обчислюємо матрицю Якобіана в заданій точці
jacobian, jacobian_at_point = jacobian_matrix(funcs, vars, point)

# Виводимо результат
print("Матриця Якобіана :\n", np.array(jacobian))
print("Матриця Якобіана у точці", point, ":\n", jacobian_at_point)
