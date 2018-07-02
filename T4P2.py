##Alumnos: Luis Felipe García, Genaro Laymuns, Frane Suzanic##  
##Parte c:

from gurobipy import *
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy
import time

try:
	numpy.random.seed(1)
	n = 5              										# productos
	I = range(n)
	m = 10              									# partes o piezas
	J = range(m)
	k = 10     	        									# realizaciones
	K = range(k)

	c = 0.1*numpy.random.randint(10,40,m)					# costo unitario por pieza
	s = 0.9*c 												# valor de salvación de piezas
	a = numpy.random.randint(5,20,(n,m))					# matriz de consumos del producto i por la pieza j
	l = numpy.random.randint(20,80,n)						# precio de adquisición del producto
	q = 8*l													# precio de venta del producto
	p = numpy.random.dirichlet(numpy.ones(k), size=1)[0]    # propapilidades de escenarios
	rho = 0.5												# parámetro de función objetivo

	# Modelo determinístico equivalente
	fde = Model()
	fde.Params.OutputFlag = 0
	x = fde.addVars(J, name="x")				#  Cantidad de piezas a comprar
	z = fde.addVars(K,I, name="z")				#  Cantidad de productos terminados a complementar en cada escenario
	y = fde.addVars(K,J, name="y")				#  Cantidad de piezas a revender en cada escenario
	# Variables auxiliares
	w = fde.addVars(K, name="w")					#  Máximo entre F(x,\xi^k)-E[F(x,\xi)] y 0
	v = fde.addVars(K, lb=-GRB.INFINITY, name="v")	#  Costo de F(x,\xi^k)	(variable sin restricción de signo)
	u = fde.addVar(lb=-GRB.INFINITY, name="u")		#  Costo promedio E[F(x,\xi)]
	# Restricciones
	fde.addConstrs(quicksum(a[i][j]*z[k,i] for i in I)+y[k,j] == x[j] for j in J for k in K)
	for k in K:										# Restricciones de demanda estocásticas:
		d = numpy.random.uniform(low=25.0, high=100.0, size=m)
		fde.addConstrs(z[k,i] <= d[i] for i in I)
	fde.addConstrs(v[k] ==	(	 quicksum(c[j]*x[j] for j in J)
								+quicksum((l[i]-q[i])*z[k,i] for i in I)
								-quicksum(s[j]*y[k,j] for j in J)) for k in K)
	fde.addConstr(u == quicksum(p[k]*v[k] for k in K))
	fde.addConstrs(w[k] >= v[k]-u for k in K)

	# Función objetivo
	fde.setObjective(u+rho*quicksum(p[k]*w[k] for k in K), GRB.MINIMIZE)
	# Solución:
	fde.optimize()
	print("Valor Objetivo: ", fde.objVal)
	# x solución fijo:
	x_bar = [];
	print('Solución x fijo:')
	for j in J:
		print(x[j].varName, x[j].x)
		x_bar.append(x[j].x)

except GurobiError:
	print('Error reported')