import sympy
x1 = sympy.symbols('x1')
x2 = sympy.symbols('x2')
x3 = sympy.symbols('x3')

def f(x1, x2, x3):
    return 3*(x1**2 + x2*x3)

u = f(x1, x2, x3)
print('f(x1, x2, x3) = ',u)

gradient_fun = [u.diff(x1), u.diff(x2), u.diff(x3)]
print('gradient function =', gradient_fun)

gradient_val = [g.subs({x1: 2, x2: 3, x3: 4}) for g in gradient_fun]
print('gradient values =', gradient_val)
