##Alumnos: Luis Felipe García, Genaro Laymuns, Frane Suzanic##  
##Parte c:

from gurobipy import *
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy
import time

try:
	numpy.random.seed(1)
	n = 5              							# productos
	I = range(n)
	m = 10              						# partes o piezas
	J = range(m) 
	c = 0.1*numpy.random.randint(10,40,m)		# costo unitario por pieza
	s = 0.9*c 									# valor de salvación de piezas
	a = numpy.random.randint(5,20,(n,m))		# matriz de consumos del producto i por la pieza j
	l = numpy.random.randint(20,80,n)			# precio de adquisición del producto
	q = 8*l										# precio de venta del producto

	### Solución fija ###
	N = 5											# cantidad de realizaciones promediadas en la SAA
	M = 5											# cantidad de simulaciones SAA iid

	k = N      	        							# realizaciones
	K = range(k)
	# Modelo determinístico equivalente
	fde = Model()
	fde.Params.OutputFlag = 0
	x = fde.addVars(J, name="x")				#  Cantidad de piezas a comprar
	z = fde.addVars(K,I, name="z")				#  Cantidad de productos terminados a complementar en cada escenario
	y = fde.addVars(K,J, name="y")				#  Cantidad de piezas a revender en cada escenario
	fde.setObjective(quicksum(c[j]*x[j] for j in J)
		+(1/N)*quicksum(quicksum((l[i]-q[i])*z[k,i] for i in I)-quicksum(s[j]*y[k,j] for j in J) for k in K)
		, GRB.MINIMIZE)
	fde.addConstrs(quicksum(a[i][j]*z[k,i] for i in I)+y[k,j] == x[j] for j in J for k in K)

	# Restricciones de demanda estocásticas:
	for k in K:
		d = numpy.random.uniform(low=25.0, high=100.0, size=m)
		fde.addConstrs(z[k,i] <= d[i] for i in I)

	# Solución:
	fde.optimize()
	print("Valor Objetivo: ", fde.objVal)
	# x solución fijo:
	x_bar = [];
	print('Solución x fijo:')
	for j in J:
		print(x[j].varName, x[j].x)
		x_bar.append(x[j].x)

	### Estimaciones f_N(x_bar) y sigma_N^2 ###
	N = 100
	k = N      	        							# realizaciones
	K = range(k)
	F_estim = []									# F(x_bar, chi_k)
	for k in K:
		k = 1										# realizaciones
		K = range(k)
		# Modelo determinístico equivalente
		fde = Model()
		fde.Params.OutputFlag = 0
		x = x_bar									#  Cantidad de piezas a comprar
		z = fde.addVars(I, name="z")				#  Cantidad de productos terminados a complementar en cada escenario
		y = fde.addVars(J, name="y")				#  Cantidad de piezas a revender en cada escenario
		fde.setObjective(quicksum(c[j]*x[j] for j in J)
			+quicksum((l[i]-q[i])*z[i] for i in I)-quicksum(s[j]*y[j] for j in J)
			, GRB.MINIMIZE)
		fde.addConstrs(quicksum(a[i][j]*z[i] for i in I)+y[j] == x[j] for j in J)
		
		# Restriccion de demanda estocástica:
		d = numpy.random.uniform(low=25.0, high=100.0, size=m)
		fde.addConstrs(z[i] <= d[i] for i in I)

		# Solución:
		fde.optimize()
		F_estim.append(fde.objVal)
	f_N = numpy.mean(F_estim)
	sigma_N = numpy.sqrt((1/N)*numpy.var(F_estim,ddof=1))

	### Estimación v_NM y sigma_NM^2 ###
	N = 100;
	M = 100;
	sol = [];
	k = N      	        						# realizaciones
	K = range(k)
	for realiz in range(M):						# simulaciones SAA iid				
	    # Modelo determinístico equivalente
		fde = Model()
		fde.Params.OutputFlag = 0
		x = fde.addVars(J, name="x")				#  Cantidad de piezas a comprar
		z = fde.addVars(K,I, name="z")				#  Cantidad de productos terminados a complementar en cada escenario
		y = fde.addVars(K,J, name="y")				#  Cantidad de piezas a revender en cada escenario
		fde.setObjective(quicksum(c[j]*x[j] for j in J)
			+(1/N)*quicksum(quicksum((l[i]-q[i])*z[k,i] for i in I)-quicksum(s[j]*y[k,j] for j in J) for k in K)
			, GRB.MINIMIZE)
		fde.addConstrs(quicksum(a[i][j]*z[k,i] for i in I)+y[k,j] == x[j] for j in J for k in K)

		# Restricciones de demanda estocásticas:
		for k in K:
			d = numpy.random.uniform(low=25.0, high=100.0, size=m)
			fde.addConstrs(z[k,i] <= d[i] for i in I)

		# Solución:
		fde.optimize()
		print("Valor Objetivo: ", fde.objVal)
		sol.append(fde.objVal)
	v_NM = numpy.mean(sol)
	sigma_NM = numpy.sqrt((1/M)*numpy.var(sol,ddof=1))

	### Estimación gap ###

	for alpha in [0.95, 0.99]:							# porcentaje de confianza
		z_alpha = st.norm.ppf(alpha)					# valor crítico a nivel alpha de Normal(0,1) 
		gap_est = f_N - v_NM + z_alpha*(numpy.sqrt(numpy.power(sigma_N,2)+numpy.power(sigma_NM,2)))
		print('% Confianza: ', alpha, '    gap estimado: ', gap_est)


except GurobiError:
	print('Error reported')