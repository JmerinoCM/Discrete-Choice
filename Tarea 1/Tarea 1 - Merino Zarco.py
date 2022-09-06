#!/usr/bin/env python
# coding: utf-8

# # Tarea 1, Elección Discreta
# ### Juan José Merino Zarco 

# ### Pregunta 1 Logit condicional (50 puntos)

# In[146]:


# Importar las librerias necesarias
import pandas as pd
import numpy as np
from scipy import optimize
from sympy import symbols, Matrix, Transpose
from math import e, log, exp
from scipy.optimize import minimize
import math
import numdifftools as nd
import os


# In[ ]:


os.chdir("D:/Usuario/Desktop/Cuarto Semestre/Elección discreta/Tarea 1") #Definir carpeta de trabajo


# In[147]:


# Importar la base de datos
yogurth = pd.read_csv("yogurt.csv")


# In[148]:


yogurth.head() #Analizamos las variables de la base de datos


# In[149]:


def modelo_yogurth(x):
    """Logit condicional usando los datos en yogurt.csv."""
    alph1,alph2,alph3,bp,bf = x
    # Valores vacios
    num = 0
    lden = 0
    calc = 0
    alph4 = 0
    alphas=[alph1, alph2, alph3, alph4]
    
    for index, row in yogurth.iterrows():
        #Numerador (realizamos las operaciones correspondientes para el númerador de la ecuación a maximizar)
        for producto in range(4):
            num = num + row[5+producto]* (alphas[producto]+ bp*row[13+producto] + bf*row[9+producto])
        
        
        #Denominador (realizamos las operaciones correspondientes para el denominador de la ecuación a maximizar)
        den = 0
        for producto in range(4):
            den = den + e**(alphas[producto]+ bp*row[13+producto] + bf*row[9+producto])
       
        # Aplicamos el resto de la función, junto a la sumatoria faltante
        for producto in range(4):
            lden = lden + row[5+producto]* log(den)
            
    #Realizamos el cálculo de la función que nos dará la ecuación completa y al final se multiplica por -1 para que sea una maximización
    calc = calc + (num-lden)
    return (calc*-1)


# ### Corremos varios modelos para robustez 

# In[201]:


#Corremos el modelo con el método de optimización BFGS
x0 = [1,1,1,1,1]
optimize.minimize(modelo_yogurth, x0, method = "BFGS") 


# In[197]:


#Corremos el modelo con el método de optimización Nelder Mead
x0 = [1,1,1,1,1]
optimize.minimize(modelo_yogurth, x0, method = "Nelder-Mead") 


# In[198]:


#Corremos el modelo con el método de optimización L-BFGS-B
x0 = [1,1,1,1,1]
optimize.minimize(modelo_yogurth, x0, method = "L-BFGS-B") 


# ### Modelo final (método BFGS)

# In[173]:


x0 = [1,1,1,1,1]  #Establecemos el punto inicial para la optimización
modelo = optimize.minimize(modelo_yogurth, x0, method = "Nelder-Mead")  #Terminación exitosa


# In[175]:


# Obtenemos los coeficientes estimados 
betas_hat = modelo["x"]
betas_hat


# In[176]:


# Errores estándar
# Iniciamos un proceso de varios pasos en código para obtener los errores estándar de los estimadores 
hessiano_inv = modelo["hess_inv"]  #Obtenemos la inversa del Hessiano evaluado en el óptimo. Esto es dado por la optimización.
hessiano_inv


# In[177]:


hessiano = np.linalg.inv(hessiano_inv)  #Sacamos la inversa del hessiano inverso para tener el hessiano original
hessiano


# In[178]:


I =  np.dot((-1/2430), hessiano) #Hacemos operaciones para obtener el valor I y posteriormente la matriz de varianzas y covarianzas
I


# In[179]:


mat_var_cov = np.linalg.inv(I)  #Aplicamos la matriz inversa a I y así obtenemos la matriz de varianzas y covarianzas
mat_var_cov = np.dot(-1,mat_var_cov)  #Multiplicamos por -1 dado que la función original la habíamos multiplicado por -1.
mat_var_cov


# In[180]:


np.shape(mat_var_cov)


# In[181]:


# Errores estándar en orden (alpha1, alpha2, alpha3, beta_price, beta_feat)
for i in range(5):
    e_e = (math.sqrt(mat_var_cov[i,i]))  / (math.sqrt(2430))
    print(e_e) 


# In[183]:


# Valor de la máxima verosimilitud 
mv_betas = - modelo_yogurth(modelo["x"])
mv_betas


# In[184]:


# Valor para las coeficientes en cero 
mv_ceros = - modelo_yogurth((0,0,0,0,0))
mv_ceros


# In[185]:


# Índice de razón de verosimilitud 
razon_vero = 1 - (mv_betas / mv_ceros)    
razon_vero


# In[186]:


# Criterior de información de Akaike (AIC) 
aic = (-2 * mv_betas) + (2*5)     #5 parametros 
aic


# In[7]:


# Elasticidades 
yogurth.head()


# In[187]:


def probabilidad_eles(dataframe):
    """Logit condicional usando los datos en yogurt.csv."""
    # Importar la base de datos
    moi = pd.read_csv(dataframe)
    #Parametros
    # alph1,alph2,alph3,bp,bf = x
    #Evaluando los parametros en el optimo
    alph1 = 1.38796
    alph2 = 0.643527
    alph3 = -3.086088
    alph4 = 0
    bp    = -37.0679
    bf    = 0.4876 
    # Valores vacios
    num = 0
    lden = 0
    calc = 0
    conca1, conca2, conca3, conca4= [],[],[],[]
    alphas=[alph1, alph2, alph3, alph4]
    for index, row in moi.iterrows():

        #Denominador
        den = 0
        for producto in range(4):
            den += exp(alphas[producto]+ bp*row[13+producto] + bf*row[9+producto])
       
        # Probabilidad
        proby1 = 0
        proby2 = 0
        proby3 = 0
        proby4 = 0
        for producto in range(4):
            
            num = exp(alphas[producto]+ bp*row[13+producto] + bf*row[9+producto])  #Numerador de la ecuación
            
            if producto == 0:         #Creamos 4 columnas (variables) a partir de una condición para cada probabilidad de cada producto
                proby1 += (num)/den
     
            elif producto == 1:
                proby2 += (num)/den
     
            elif producto == 2:
                proby3 += (num)/den
       
            else:
                proby4 += (num)/den
    
                     
        conca1.append(proby1)   #Con este codigo logramos generar las columnas finales de cada probabilidad de cada producto
        conca2.append(proby2)
        conca3.append(proby3)
        conca4.append(proby4)
        
    base = pd.concat([moi,pd.DataFrame(conca1)],axis=1)   #Unimos cada columna en un solo dataframe
    base.rename({0: 'proba1'}, axis=1, inplace=True)
    base = pd.concat([base,pd.DataFrame(conca2)],axis=1)
    base.rename({0: 'proba2'}, axis=1, inplace=True)
    base = pd.concat([base,pd.DataFrame(conca3)],axis=1)
    base.rename({0: 'proba3'}, axis=1, inplace=True)
    base = pd.concat([base,pd.DataFrame(conca4)],axis=1)
    base.rename({0: 'proba4'}, axis=1, inplace=True)
    
    
    #Calculo
    return base #La función devuelve la base final con las 4 columnas de probabilidad 


# In[190]:


elas = probabilidad_eles("yogurt.csv")  #Generamos un dataframe uniendo a yogurth con las columnas de probabilidad


# In[191]:


elas.insert(22, "Bz", -37.0679, allow_duplicates=False)   #Creamos una columna con el valor estimado de beta_price (esta es la derivada)


# In[192]:


##Los valores de la probabilidad, los precios y la beta_price los cambiamos a formato float
elas["proba1"] = elas["proba1"].astype(float, errors = 'raise')
elas["proba2"] = elas["proba2"].astype(float, errors = 'raise')
elas["proba3"] = elas["proba3"].astype(float, errors = 'raise')
elas["proba4"] = elas["proba4"].astype(float, errors = 'raise')
elas["Bz"] = elas["Bz"].astype(float, errors = 'raise')
elas["price1"] = elas["price1"].astype(float, errors = 'raise')
elas["price2"] = elas["price2"].astype(float, errors = 'raise')
elas["price3"] = elas["price3"].astype(float, errors = 'raise')
elas["price4"] = elas["price4"].astype(float, errors = 'raise')


# In[193]:


# Elasticidades propias
# Procedemos a calcular las elasticidades propias creando 4 columnas por cada elasticidad para todas las filas
elas["elas_prop1"] = elas["Bz"] * elas["price1"] * (1-elas["proba1"])
elas["elas_prop2"] = elas["Bz"] * elas["price2"] * (1-elas["proba2"])
elas["elas_prop3"] = elas["Bz"] * elas["price3"] * (1-elas["proba3"])
elas["elas_prop4"] = elas["Bz"] * elas["price4"] * (1-elas["proba4"])


# In[194]:


# Elasticidades propias
# Procedemos a calcular la media de cada elasticidad (columnas creadas anteriormente) y así tener la elasticidad promedio final.
mean_elas_prop1 = elas["elas_prop1"].mean()
mean_elas_prop2 = elas["elas_prop2"].mean()
mean_elas_prop3 = elas["elas_prop3"].mean()
mean_elas_prop4 = elas["elas_prop4"].mean()
print(mean_elas_prop1, mean_elas_prop2,mean_elas_prop3,mean_elas_prop4)


# In[195]:


# Elasticidades cruzadas
# Procedemos a calcular las elasticidades cruzadas creando 4 columnas por cada elasticidad para todas las filas
elas["elas_cru21"] = elas["Bz"] * elas["price1"] * (-elas["proba1"])
elas["elas_cru12"] = elas["Bz"] * elas["price2"] * (-elas["proba2"])
elas["elas_cru13"] = elas["Bz"] * elas["price3"] * (-elas["proba3"])
elas["elas_cru14"] = elas["Bz"] * elas["price4"] * (-elas["proba4"])


# In[196]:


# Elasticidades cruzadas
# Procedemos a calcular la media de cada elasticidad (columnas creadas anteriormente) y así tener la elasticidad promedio final.
mean_elas_cru1 = elas["elas_cru21"].mean()
mean_elas_cru2 = elas["elas_cru12"].mean()
mean_elas_cru3 = elas["elas_cru13"].mean()
mean_elas_cru4 = elas["elas_cru14"].mean()
print(mean_elas_cru1, mean_elas_cru2,mean_elas_cru3,mean_elas_cru4)


# ## Pregunta 2: Fake data (50 puntos)

# In[202]:


import pandas as pd
import numpy as np
import random
from scipy import optimize, stats
from math import e, log,exp
import math as mt
import matplotlib.pyplot as plt 
from scipy.stats import bernoulli


# ### 1. Generación de 100 bases de datos con 5000 observaciones cada una 
# 
# Se generaran datos pseudo-aleatorios , empleando una distribución de Bernoulli para construir la característica "feat", una distribución log-normal para los precios de  los 4 productos para las N observaciones.

# In[209]:


"""Creación de las bases de datos"""
# Parametros de la función
## N = Número de datos, ##J= Número de bases
N=5000
J=100
# Precios promedio
yogurt = pd.read_csv("yogurt.csv")
precios = yogurt[['price1', 'price2', 'price3', 'price4']]
precios_promedio = list(precios.mean())

# Valores de los parámetros
beta_p_s=np.array([1,-1,0.5,0,-20,2])

#  Valores vacios
base_precios = pd.DataFrame(np.nan, index = range(N), columns = ['price1', 'price2', 'price3', 'price4'])
atributos = pd.DataFrame(np.nan, index = range(N), columns = ['feat1', 'feat2', 'feat3', 'feat4'])
dict_data_frames = {}

# Creación de las bases
bases = [str("base") + str(x) for x in range(J)]

# Generar las características en cada base
for base in bases:
    for j in range(4): 
        base_precios.iloc[:,j] = np.random.lognormal(precios_promedio[j],0.6,N)
        atributos.iloc[:,j] = bernoulli.rvs(size = N,p = 0.5)
        df = pd.concat([base_precios,atributos], axis=1)

    dict_data_frames[base] = df 


# Loop para simular la elección de los consumidores
for base in bases:

    ele1,ele2,ele3,ele4 = [],[],[],[]

    for index, row in dict_data_frames[base].iterrows():
        s_1 = np.array([1,0,0,0,row[0],row[4]])
        s_2 = np.array([0,1,0,0,row[1],row[5]])
        s_3 = np.array([0,0,1,0,row[2],row[6]])
        s_4 = np.array([0,0,0,1,row[3],row[7]])

        for i in [1,2,3,4]: 
            globals()['exp_%s' %i] = mt.exp(np.dot(globals()['s_%s' %i],beta_p_s))    #iterar por nombres

        total_exp = exp_1+exp_2+exp_3+exp_4

        for j in [1,2,3,4]:
            globals()['prob_%s' %j] = (globals()['exp_%s' %j]) / (total_exp)

        eleccion=np.random.choice((1,2,3,4),p=[prob_1,prob_2,prob_3,prob_4])

        if eleccion==1:
            ele1.append(1)
            ele2.append(0)
            ele3.append(0)
            ele4.append(0)

        elif eleccion==2:
            ele1.append(0)
            ele2.append(1)
            ele3.append(0)
            ele4.append(0)
        elif eleccion==3:
            ele1.append(0)
            ele2.append(0)
            ele3.append(1)
            ele4.append(0)
        else:
            ele1.append(0)
            ele2.append(0)
            ele3.append(0)
            ele4.append(1)

    dict_data_frames[base]["brand1"]=ele1 
    dict_data_frames[base]["brand2"]=ele2
    dict_data_frames[base]["brand3"]=ele3  
    dict_data_frames[base]["brand4"]=ele4 
    
# Guardo el diccionario de todas las bases de datos utilizado en un objeto

fin = dict_data_frames
          


# Procedemos a incorporar las bases de datos simuladas en la función para evaluar el modelo de logit condicional

# In[210]:


def modelo_yogurth_MC(x,diccionario,base_de_datos):
    """Logit condicional usando los datos de la base proxy"""
    #Parametros desconocidos
    [alph1,alph2,alph3,bp,bf] = x 
    #Normalizar \alpha_{4} = 0
    alph4 = 0
    # Valores vacios
    lden = 0
    # Importar la base de datos
    # diccionario = el diccionario que contiene las 100 bases
    # base_de_datos = base contenia en el diccionario
    dict_data_frames = diccionario
    data = dict_data_frames[base_de_datos]
    # Variables para cada producto
    alphas =[alph1,alph2,alph3,alph4] 
    precios = ["price1","price2","price3","price4"]
    Yin = ["brand1","brand2","brand3","brand4"]
    feat = ["feat1","feat2","feat3","feat4"]
    
    for index, row in data.iterrows():

        #Denominador
        den = 0
        for producto in range(4):
            den += exp(alphas[producto]+ bp*row[precios[producto]] + bf*row[feat[producto]])
       
        # Logaritmo natual de la suma
        for producto in range(4):
            if den == 0:
                continue
            else:
                #Numerador
                lden += row[Yin[producto]]* log(exp(alphas[producto]+ bp*row[precios[producto]] + bf*row[feat[producto]])/den)
                moly = -lden
    return moly


# Ahora, vamos a realizar la optimizacion para las 100 bases de datos mediante la siguiente función

# In[211]:


def Monte_carlo():
    conca = pd.DataFrame()
    for index, (key, value) in enumerate(dict_data_frames.items()):
        print(index) # Control de tiempo
        result = optimize.minimize(modelo_yogurth_MC, x0=[0,0,0,0,0],args=(fin,key))
        conca = pd.concat([conca,pd.DataFrame(result.x)],axis=1)
    conca = pd.DataFrame.transpose(conca)   
    conca.rename({0: 'alph1',1: 'alph2',2: 'alph3',3: 'bp',4: 'bf'}, axis=1, inplace=True)
    conca["alph4"] = 0
    conca.reset_index(drop=True, inplace=True)
    return conca


# Se guardan los resultados en un archivo csv para la posterior utilización de los datos

# In[ ]:


MC_save = pd.DataFrame(Monte_carlo())
MC_save.to_csv("MC_save.csv")


# ### Mediana de los estimadores para cada uno de los 5 coeficientes.

# In[212]:


# Cargamos los datos
base_B = pd.read_csv("MC_save.csv")
print(pd.DataFrame.median(base_B))


# ### Histograma con la distribución de los estimadores para cada coeficiente.

# In[213]:


base_B.drop(base_B.columns[0], axis=1, inplace=True)


# In[214]:


base_B.hist(layout=(3,2),figsize=(8,8),grid=False,color="#800080",zorder=2, rwidth=0.9)


# ## Fin 
