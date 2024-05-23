import sympy as sp

# Визначення символьної змінної
x = sp.symbols('x')

# Визначення функції експоненти та її розклад у ряд Тейлора
exp_func = sp.exp(-x**2)
taylor_series_exp = sp.series(exp_func, x, 0, 15).removeO() # Ряд Тейлора до 5-го члена

# Обчислення інтеграла за допомогою ряду Тейлора
integral_taylor_exp = sp.integrate(taylor_series_exp, (x, 0, 1))

# Вивід результатів
print("Функція експоненти:", exp_func)
print("Ряд Тейлора для експоненти:", taylor_series_exp)
print("Інтеграл за допомогою ряду Тейлора:", integral_taylor_exp.evalf())
