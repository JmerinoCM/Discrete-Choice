# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:10:30 2022
Tarea 4
@author: jujo_
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize,fsolve
import os
from numba import njit,prange
from numba.typed import List
import collections 
import time


#Working directory
os.chdir('C:\Discrete Choice\Tarea 4 V2')

#Import raw data
df=pd.read_csv("car_data.csv")
df_nodes_d5 = np.genfromtxt("GQU_d5_l4.csv", delimiter=',', skip_header=1)

#Seleccionar columnas de interes
df2 = df[["id","air","car_size","mpd","p_real","hp2wt"]]

#Crear dataframe con modelos de carro unicos
df_unique = df2.drop_duplicates(subset = "id")
##ordenar productos
df_unique = df_unique.sort_values(by=["id"],ascending=True)
df_unique_np = np.asarray(df_unique)

# Cuantas veces se compro cada producto
df_count = df.sort_values(by=["id"],ascending=True)
c1 = collections.Counter(df_count["id"])
# Convertir en array para que sea compatible con numba
c1_np = np.array(list(c1.items()))

#Utilidades
@njit(fastmath=True , parallel=True)
def matrix(x):

    df_nodes = np.transpose(df_nodes_d5[:,:5])
    df_inter_day   = df_unique_np[:,1:6]
    
    df_day = np.zeros( (len(df_unique_np[:,1]), 5 ) )
    a = np.zeros(( len(df_unique_np[:,1]) , 1))

    df_day[:,0] =         df_inter_day[:,0]*x[0] #air
    df_day[:,1] =         df_inter_day[:,1]*x[1] #car_size
    df_day[:,2] =         df_inter_day[:,2]*x[2] #mpd
    df_day[:,3] = -       np.log(df_inter_day[:,3])*x[3] #p_real
    df_day[:,4] =         df_inter_day[:,4]*x[4] #hp2wt
    
    a[:,0] = np.asarray(x[-(len(x)-5):])         
          
    v_n = np.exp(a + np.dot(df_day,df_nodes))
      
    return v_n

## logit
@njit(fastmath=True , parallel=True)
def mixed_ll(x):
    v_n = matrix(x)
    #denominador
    den = np.ones((len(df_nodes_d5[:,1]),1)) # 1 por la outside option
    for node in prange(len(df_nodes_d5[:,1])):
        v_n_den = v_n[:,node]
        den[node,0] += np.sum(v_n_den)
    
    #probabilidades productos in
    s = np.zeros((len(df_unique_np[:,1]),1))
    for producto in prange(len(df_unique_np[:,1])):
        frac = 0
        for node in range(len(df_nodes_d5[:,1])):
            num = v_n[producto,node]
            frac += (num/den[node,0])*df_nodes_d5[node,5]
        s[producto,0] = frac
    s = s[s[:,0] > 0]
    #probabilidad de la outside option
    frac_out = 0
    for node in prange(len(df_nodes_d5[:,1])):
        num = 1
        frac_out += (num/den[node,0])*df_nodes_d5[node,5]
    s_outside = frac_out
    
    #calculo de la likelihood
    l = np.zeros((len(df_unique_np [:,1]) + 1 , 1)) # +1 por la outside
    #inside options
    for prod in prange(len(df_unique_np[:,1])):
        l[prod,0] = np.log(s[prod,0])*c1_np[prod,1]
    #outside options(cuantos compraron la outside)
    out_n = 6
    l[-1,0] = np.log(s_outside)*out_n
    return  -np.sum(l)

# Guardar datos
save_op = pd.DataFrame()

# construccion del punto inicial
deltas = np.zeros(len(df_unique_np [:,1]))
caracteristicas = [0,0,0,0,0]
coeficientes = np.append( caracteristicas, deltas)
x0 = List(coeficientes)

#La velocidad de numba se nota hasta despues de compilar una vez la funcion
start = time.time()
mixed_ll(x0)
end = time.time()
print("Tiempo evaluacion mixed logit/first = %s" % (end - start))

start = time.time()
mixed_ll(x0 )
end = time.time()
print("Tiempo evaluacion mixed logit/second = %s" % (end - start))

#Optimizacion de la funcion con diferentes metodos

#### usando x0
methods= ["Nelder-Mead","BFGS","L-BFGS-B"]
for metodo in methods:
    g = pd.DataFrame(index=[0])
    start = time.time()
    temp = minimize(mixed_ll, x0 = x0,method = metodo)
    end = time.time()
    g["Metodo"] = metodo
    g["Tiempo"] = time.strftime("%s" % (end - start))
    g["Exito de la optimizacion"] = temp["success"] 
    g["Número de iteraciones"] = temp["nit"]
    g["Valor de la función en el óptimo"] = temp["fun"]
    g["Punto inicial para todas las variables"] = "0"
    save_op = pd.concat([save_op,g],axis=0)
    save_op.reset_index(drop=True, inplace=True)

## Recuperamos(y guardamos) el punto optimo
print("punto optimo")
b = minimize(mixed_ll, x0 = x0,method = "L-BFGS-B")
print("Success = ", b["success"])

x_opt = b["x"]   
df_x_opt = pd.DataFrame(x_opt)

try:
    os.remove('t4_val_optimos.csv')       
except:
    pass 
df_x_opt.to_csv('t4_val_optimos.csv', mode='a', index=False, header=True)


## probamos puntos iniciales distintos 

### construccion del punto inicial
deltas_o = np.ones(len(df_unique_np [:,1]))
caracteristicas_o = [1,1,1,1,1]
coeficientes = np.append( caracteristicas_o, deltas_o)
x0_o = List(coeficientes)

methods= ["Nelder-Mead","BFGS","L-BFGS-B"]
for metodo in methods:
    start = time.time()
    temp_o = minimize(mixed_ll, x0 = x0_o,method = metodo)
    end = time.time()
    g["Metodo"] = metodo
    g["Tiempo"] = time.strftime("%s" % (end - start))
    g["Exito de la optimizacion"] = temp["success"] 
    g["Número de iteraciones"] = temp["nit"]
    g["Valor de la función en el óptimo"] = temp["fun"]
    g["Punto inicial para todas las variables"] = "1"
    save_op = pd.concat([save_op,g],axis=0)
    save_op.reset_index(drop=True, inplace=True)

### construccion del punto inicial
deltas_m = np.full((len(df_unique_np [:,1])),-1)
caracteristicas_m = [-1,-1,-1,-1,-1]
coeficientes = np.append( caracteristicas_m, deltas_m)
x0_m = List(coeficientes)

methods= ["Nelder-Mead","BFGS","L-BFGS-B"]
for metodo in methods:
    start = time.time()
    temp_m = minimize(mixed_ll, x0 = x0_m,method = metodo)
    end = time.time()
    g["Metodo"] = metodo
    g["Tiempo"] = time.strftime("%s" % (end - start))
    g["Exito de la optimizacion"] = temp["success"] 
    g["Número de iteraciones"] = temp["nit"]
    g["Valor de la función en el óptimo"] = temp["fun"]
    g["Punto inicial para todas las variables"] = "-1"
    save_op = pd.concat([save_op,g],axis=0)
    save_op.reset_index(drop=True, inplace=True)

try:
    os.remove('optimizacion.csv')       
except:
    pass 
save_op.to_csv('optimizacion.csv', mode='a', index=False, header=True)


# Construccion de instrumentos

df_inst = pd.DataFrame(df[["id","firm_id","air","car_size","mpd","hp2wt","p_real"]])

### BLP IV 3
ive = pd.DataFrame()
ive[["air_ive","car_size_ive","mpd_ive","hp2wt_ive"]] =\
    df_inst.groupby("firm_id")[["air","car_size","mpd","hp2wt"]].sum().sum()\
        - df_inst.groupby("firm_id")[["air","car_size","mpd","hp2wt"]].sum()
        
df_inst = df_inst.merge(ive, left_on='firm_id', right_on='firm_id')

### BLP IV 2
firm_own = pd.DataFrame()
firm_own[["air_own","car_size_own","mpd_own","hp2wt_own"]] =\
    df_inst.groupby("firm_id")[["air","car_size","mpd","hp2wt"]].sum()

df_inst = df_inst.merge(firm_own, left_on='firm_id', right_on='firm_id')
        
df_inst["air_ivi"] = df_inst["air_own"] - df_inst["air"]
df_inst["car_size_ivi"] = df_inst["car_size_own"] - df_inst["car_size"]
df_inst["mpd_ivi"] = df_inst["mpd_own"] - df_inst["mpd"]
df_inst["hp2wt_ivi"] = df_inst["hp2wt_own"] - df_inst["hp2wt"]


df_inst_unique = df_inst.drop_duplicates(subset = "id")
df_inst_unique = df_inst_unique.sort_values(by=["id"],ascending=True)

# agregamos las deltas

df_inst_unique["deltas"] = x_opt[:-5]

try:
    os.remove('instrumentalizacion.csv')       
except:
    pass 
df_inst_unique.to_csv('instrumentalizacion.csv', mode='a', index=False, header=True)

######## Estimación de Variables instrumentales por el método de momentos

df_solve = pd.read_csv("instrumentalizacion.csv")

df_solve["log_p_real"]=np.log(df_solve["p_real"])

## Solución sistema de ecuaciones con 1 solo instrumento 

def solve_function(solved_value):
    
    beta1, beta2, beta3, beta4, alphap = solved_value[0], solved_value[1], solved_value[2], solved_value[3], solved_value[4]
    
    f1 = (1/109)*df_solve['air_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"]) 
    
    f2 = (1/109)*df_solve['hp2wt']*(df_solve['deltas']-(beta1*df["hp2wt"]+beta2*df["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f3 = (1/109)*df_solve['air']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f4 = (1/109)*df_solve['mpd']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f5 = (1/109)*df_solve['car_size']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])    
    
    #Sumar sobre todas las observaciones
    fila1 = np.sum(f1)
    fila2 = np.sum(f2)
    fila3 = np.sum(f3)
    fila4 = np.sum(f4)
    fila5 = np.sum(f5)
    
    return [fila1, fila2, fila3, fila4, fila5]

solved = fsolve(solve_function,[0, 0, 0, 0, 0])

print(solved)

## Solución sistema de ecuaciones con 2 instrumentos

def solve_function2(solved_value):
    
    beta1, beta2, beta3, beta4, alphap = solved_value[0], solved_value[1], solved_value[2], solved_value[3], solved_value[4]
    
    f1 = (1/109)*((df_solve['air_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['mpd_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"]))) 
    
    
    f2 = (1/109)*df_solve['hp2wt']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"] \
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f3 = (1/109)*df_solve['air']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"] \
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f4 = (1/109)*df_solve['mpd']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"] \
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f5 = (1/109)*df_solve['car_size']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"] \
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])    
    
    #Sumar sobre todas las observaciones
    fila1 = np.sum(f1)
    fila2 = np.sum(f2)
    fila3 = np.sum(f3)
    fila4 = np.sum(f4)
    fila5 = np.sum(f5)
    
    return [fila1, fila2, fila3, fila4, fila5]

solved2 = fsolve(solve_function2,[0, 0, 0, 0, 0])

print(solved)
print(solved2)


## Solución sistema de ecuaciones con 3 instrumentos

def solve_function3(solved_value):
    
    beta1, beta2, beta3, beta4, alphap = solved_value[0], solved_value[1], solved_value[2], solved_value[3], solved_value[4]
    
    f1 = (1/109)*((df_solve['air_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['mpd_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['hp2wt_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"]))) 
    
    
    f2 = (1/109)*df_solve['hp2wt']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f3 = (1/109)*df_solve['air']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f4 = (1/109)*df_solve['mpd']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f5 = (1/109)*df_solve['car_size']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])    
    
    #Sumar sobre todas las observaciones
    fila1 = np.sum(f1)
    fila2 = np.sum(f2)
    fila3 = np.sum(f3)
    fila4 = np.sum(f4)
    fila5 = np.sum(f5)
    
    return [fila1, fila2, fila3, fila4, fila5]

solved3 = fsolve(solve_function3,[0, 0, 0, 0, 0])

print(solved)
print(solved2)
print(solved3)


## Solución sistema de ecuaciones con 4 instrumentos (IVE)

def solve_function4(solved_value):
    
    beta1, beta2, beta3, beta4, alphap = solved_value[0], solved_value[1], solved_value[2], solved_value[3], solved_value[4]
    
    f1 = (1/109)*((df_solve['air_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['mpd_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['hp2wt_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['car_size_ive']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"]))) 
    
    
    f2 = (1/109)*df_solve['hp2wt']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f3 = (1/109)*df_solve['air']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f4 = (1/109)*df_solve['mpd']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f5 = (1/109)*df_solve['car_size']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])    
    
    #Sumar sobre todas las observaciones
    fila1 = np.sum(f1)
    fila2 = np.sum(f2)
    fila3 = np.sum(f3)
    fila4 = np.sum(f4)
    fila5 = np.sum(f5)
    
    return [fila1, fila2, fila3, fila4, fila5]

solved4 = fsolve(solve_function4,[0, 0, 0, 0, 0])

print(solved)
print(solved2)
print(solved3)
print(solved4)



## Solución sistema de ecuaciones con 4 instrumentos (IVI)

def solve_function5(solved_value):
    
    beta1, beta2, beta3, beta4, alphap = solved_value[0], solved_value[1], solved_value[2], solved_value[3], solved_value[4]
    
    f1 = (1/109)*((df_solve['air_ivi']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['mpd_ivi']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['hp2wt_ivi']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['car_size_ivi']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"]))) 
    
    
    f2 = (1/109)*df_solve['hp2wt']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f3 = (1/109)*df_solve['air']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f4 = (1/109)*df_solve['mpd']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f5 = (1/109)*df_solve['car_size']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]\
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])    
    
    #Sumar sobre todas las observaciones
    fila1 = np.sum(f1)
    fila2 = np.sum(f2)
    fila3 = np.sum(f3)
    fila4 = np.sum(f4)
    fila5 = np.sum(f5)
    
    return [fila1, fila2, fila3, fila4, fila5]

solved5 = fsolve(solve_function5,[0, 0, 0, 0, 0])

print(solved)
print(solved2)
print(solved3)
print(solved4)
print(solved5)




## Solución sistema de ecuaciones con 4 instrumentos (PROMEDIO)

df_solve["air_mean"] = (df_solve["air_ive"]+df_solve["air_ivi"])/2
df_solve["car_size_mean"] = (df_solve["car_size_ive"]+df_solve["car_size_ivi"])/2
df_solve["mpd_mean"] = (df_solve["mpd_ive"]+df_solve["mpd_ivi"])/2
df_solve["hp2wt_mean"] = (df_solve["hp2wt_ive"]+df_solve["hp2wt_ivi"])/2


def solve_function6(solved_value):
    
    beta1, beta2, beta3, beta4, alphap = solved_value[0], solved_value[1], solved_value[2], solved_value[3], solved_value[4]
    
    f1 = (1/109)*((df_solve['air_mean']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['mpd_mean']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['hp2wt_mean']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])) \
                  + (df_solve['car_size_mean']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"]+beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"]))) 
    
    
    f2 = (1/109)*df_solve['hp2wt']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"] \
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f3 = (1/109)*df_solve['air']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"] \
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f4 = (1/109)*df_solve['mpd']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"] \
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])
    
    f5 = (1/109)*df_solve['car_size']*(df_solve['deltas']-(beta1*df_solve["hp2wt"]+beta2*df_solve["air"] \
                +beta3*df_solve["mpd"]+beta4*df_solve["car_size"])-alphap*df_solve["log_p_real"])    
    
    #Sumar sobre todas las observaciones
    fila1 = np.sum(f1)
    fila2 = np.sum(f2)
    fila3 = np.sum(f3)
    fila4 = np.sum(f4)
    fila5 = np.sum(f5)
    
    return [fila1, fila2, fila3, fila4, fila5]

solved6 = fsolve(solve_function6,[0, 0, 0, 0, 0])

print(solved)
print(solved2)
print(solved3)
print(solved4)
print(solved5)
print(solved6)


conca = pd.DataFrame()
conca = pd.concat([conca,pd.DataFrame(pd.DataFrame(solved))],axis=0)
conca = pd.concat([conca,pd.DataFrame(pd.DataFrame(solved2))],axis=1)
conca = pd.concat([conca,pd.DataFrame(pd.DataFrame(solved3))],axis=1)
conca = pd.concat([conca,pd.DataFrame(pd.DataFrame(solved4))],axis=1)
conca = pd.concat([conca,pd.DataFrame(pd.DataFrame(solved5))],axis=1)
conca = pd.concat([conca,pd.DataFrame(pd.DataFrame(solved6))],axis=1)   

conca.columns=['Modelo 1','Modelo 2','Modelo 3', 'Modelo 4', \
                                'Modelo 5', 'Modelo 6']

try:
    os.remove('base_alphas_betas.csv')       
except:
    pass 
conca.to_csv('base_alphas_betas.csv', mode='a', index=False, header=True) 





