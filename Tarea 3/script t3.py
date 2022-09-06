# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 09:43:42 2022

@author: Campos Ortiz y Merino Zarco
"""

from math import e, exp
import scipy
from scipy import special
import numpy as np
import pandas as pd
import os
import itertools

# Working directory
os.chdir('C:\Discrete Choice\Tarea 3')

#Punto medio

## n^pm para conseguir la precision que se quiere
def f_n_pm(precision):
    #n para una precisión 
    r = int(round(((8*e)/(24*(precision)))**(1/2),0))
    return r

n_pm = f_n_pm(0.000001)

## calculo de la integral con el metodod el punto medio
def eval_int_1(n):
    # Punto medio
    h = (1-(-1))/n 
    nod = np.array([-1+h/2+i*h for i in range(n)])
    def f(x):
        return exp(x)
    vf = np.vectorize(f)
    r  = h*np.sum(vf(nod))
    return r

## Valor de la integral
int_pm = eval_int_1(n_pm)

#Integral por el metodo Gauss-legendre

## Integral
def eval_int_2(n):
    # Gauss-legendre
    nod,weight = scipy.special.roots_legendre(n)
    def f(x):
        return exp(x)
    #regresar fn vectorizada
    vf = np.vectorize(f)
    r = np.sum(np.dot(weight,vf(nod)))
    return r

## n^gl para conseguir la precision que se quiere
for n in itertools.count(start=1):
    i_n = eval_int_2(n)
    i_n_1 = eval_int_2(n+1)
    if abs(i_n - i_n_1) < 0.000001:
        n_gl = n
        break
    else:
        continue

### Valor de n^{gl}
n_gl

## Integral evaluada
int_gl = eval_int_2(n_gl)


# Montecarlo

## n maxima entre las n de los dos metodos anteriores
n_mc = max(n_pm, n_gl)

## seed

np.random.seed(2022)

## Integracion de montecarlo

def eval_int_3(n_mc):
    #Integracion de montecarlo
    mc_df = pd.DataFrame(np.random.uniform(size = (100,n_mc), low = -1, high = 1))
    r = np.mean((2/(n_mc))*np.sum(np.exp(mc_df),axis=1))
    return r 

##Valor de la integral
int_mc = eval_int_3(n_mc)

#Valor exacto de la integral
int_exacto = e - 1/e

#Comparativa de los resultados de los 3 metodos
comparativa = pd.DataFrame({(int_pm,int_gl,int_mc,int_exacto),
 (int_exacto-int_pm, int_exacto-int_gl, int_exacto-int_mc, int_exacto-int_exacto)},
                           columns=("Punto Medio","Gauss-Legendre","Monte Carlo", "Valor Exacto"))


# Inciso 4

#Dataframe
df = pd.read_csv("GQN_d5_l5.csv")

# Formular la f(x_{i})

# argumentos de la suma y del denominador
v1=df['alph1']-2*df['bp']
v2=df['alph2']-4*df['bp']
v3=df['alph3']-8*df['bp']
v4=df['alph4']-16*df["bp"]

expv1=v1.apply(np.exp)
expv2=v2.apply(np.exp)
expv3=v3.apply(np.exp)
expv4=v4.apply(np.exp)

# (denominador)
inclvalue=expv1+expv2+expv3+expv4

# numerador
prob1=expv1/inclvalue

#funcion evaluada multiplicada por el peso
l= prob1 * df["weight"]

#Sumar sobre todas las observaciones
int_inciso_4 = np.sum(l)



