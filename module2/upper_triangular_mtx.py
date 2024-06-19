import numpy as np

def upper_triangular(matrix):
    """
    Перетворює задану матрицю до верхньотрикутного вигляду за допомогою тотожних перетворень.

    Параметри:
    - matrix: numpy array, задана матриця.

    Повертає:
    - upper: numpy array, верхньотрикутна матриця, еквівалентна вхідній матриці.
    """
    upper = np.copy(matrix)
    
    rows, columns = upper.shape
    
    for j in range(columns):
        
        # Шукаємо ненульовий елемент у стовпці j, починаючи з рядка j
        for i in range(j, rows):
            if upper[i, j] != 0:
                
                # Знаходимо ненульовий елемент у стовпці і переставляємо його на позицію j
                upper[[i, j]] = upper[[j, i]]
                
                # Застосовуємо елементарні тотожні перетворення, щоб обнулити всі елементи під діагоналлю
                for k in range(i + 1, rows):
                    c = upper[k, j] / upper[j, j]
                    upper[k] -= c * upper[j]
                break

    return upper
                    
matrix = np.array([[2.0, -1.0, 0.0],
                   [-1.0, 2.0, -1.0],
                   [0.0, -1.0, 2.0]])

upper_triangular_matrix = upper_triangular(matrix)
print("Верхньотрикутна матриця:")
print(upper_triangular_matrix)       
        
if all(upper_triangular_matrix[i, i] > 0 for i in range(upper_triangular_matrix.shape[0])):
    print("Матриця є позитивно оптимальною.")
else:
    print("Матриця не є позитивно оптимальною.")
