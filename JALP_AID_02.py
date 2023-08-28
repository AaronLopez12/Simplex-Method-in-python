import numpy as np
from itertools import combinations
import os

#np.random.seed(0)
# np.random.seed(1)


class SimplexMethod:

	"""
	Clase para el método Simplex

	Inicialización:
	Matriz A: de los coeficientes de las restricciones
	Vector B: de los coeficientes de las constantes de las restricciones
	Vector C: de los coeficientes en la función objetivo

	De modo que: A·X = b donde X^T = (x1, x2,...,xn) es el vector de
	variables de decisión
	"""

	def __init__(self, A,b, c):

		self.A = A
		self.b = np.transpose(b)
		self.c = c
		

		print("Matriz de restricciones\n", self.A)
		print("\n Vector de constantes\n ", self.b)
		print("\n Vector de constantes de la función objetivo\n", self.c)


		m = A.shape[1]

		# Combinaciones de las columnas para iniciar el algoritmo
		Columnas = list(combinations(range(m), 2))

		# Seleccion de un par de columnas de las posibles 
		random_column = np.random.randint(len(Columnas))
		Beta = list(Columnas[random_column])
		

		# Proceso iterativo
		while True: 
			
			# Matrices necesarias  para la ejecución de la iteracion
			
			N_garigoleada = set(range(m)) - set(Beta)
			
			Matriz_B = self.A[:,Beta]
			
			x_B = np.dot(np.linalg.inv(Matriz_B), b)
			
			c_B = self.c[Beta]
			
			N = A[:, list(N_garigoleada)]
			
			Lambda = np.dot( np.linalg.inv(np.transpose(Matriz_B)), c_B)
			
			S_N = c[list(N_garigoleada)] - np.dot(np.transpose(N), Lambda )
			

			if np.any(S_N < 0):
				# En caso de que algún valor de la matríz S_n sea menor a cero
				negative_values = {j : True if S_N[i] < 0 else False for i,j in enumerate(list(N_garigoleada))}
				negative_values = {Llave: Valor for Llave, Valor in negative_values.items() if Valor is True}
				q = np.random.choice(list(negative_values.keys()) , replace = False)
				d = np.dot(np. linalg.inv(Matriz_B), self.A[:,q])
				
				if np.any(d <= 0):
					# Caso en el que el problema no esté acotado
					print(d)
					print("el problema no está restringido.") 
					break

				else:
					# Encontrando el indice que minimiza el cociente
					indices = [x_B[i] / d[i] for i in range(len(x_B))] 
					p = list(Beta)[np.argmin(indices)]
					Beta[Beta.index(p)] = q
					

			else:

				if np.any(x_B < 0):
					# Si el algoritmo encontró una solución que no respeta la no negatividad de 
					# las variables de decisión
					print("\nSe encontró una solución que no respeta la condición de no negatividad")
					print("Por favor ejecute de nuevo el algoritmo para encontrar otra solución")
					break

				# En caso contrario: se encuentra solución
				# restricciones factible que atiende a todas las 
				print("\nOptimo Encontrado")

				for i in range(len(Beta)):
					print("x", Beta[i], " = ", x_B[i])
				
				break


if __name__ == '__main__':

	A = np.array([[1,1,1,0], [2, 0.5, 0 ,1]])
	b = np.array([5, 8])
	c = np.array([-4, -2, 0,0])


	SimplexMethod(A,b, c)