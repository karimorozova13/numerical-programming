import sympy as sp
import numpy as np


# f(x,y)=x^3−2xy−y^6 
# в точці  (1,2)


def hessian_matrix(func, vars, point):
    n = len(vars)
    hessian = sp.zeros(n, n)

    # Створюємо символьні змінні для обчислення часткових похідних
    symbols = vars

    # Обчислюємо часткові похідні за допомогою SymPy
    for i, var1 in enumerate(symbols):
        for j, var2 in enumerate(symbols):
            # Обчислюємо часткову другу похідну
            hessian[i, j] = sp.diff(func, var1, var2)

    # Підставляємо значення змінних у матрицю Гессіана для заданої точки
    hessian_evaluated = np.array(hessian.subs({var: val for var, val in zip(symbols, point)}))

    return hessian, hessian_evaluated

# Приклад використання:

# Створюємо символьні змінні x і y
x, y = sp.symbols('x y')

# Задаємо символьну функцію
func = x**3 - 2*x*y - y**6

# Визначаємо список змінних для обчислення матриці Гессіана
vars = [x, y]

# Задаємо точку, в якій потрібно обчислити матрицю Гессіана
point = (1, 2)

# Обчислюємо матрицю Гессіана в заданій точці
hess, hess_at_point = hessian_matrix(func, vars, point)

# Виводимо результат
print("Матриця Гессіана:\n", np.array(hess))
print("Матриця Гессіана у точці", point, ":\n", hess_at_point)