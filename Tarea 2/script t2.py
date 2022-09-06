# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 00:44:10 2022

 German Campos y Juan Merino
"""

#Modules

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os
import numdifftools as nd


#Working directory
os.chdir('C:\Discrete Choice\Tarea 2')

#Import raw data
df=pd.read_csv("yogurt.csv")


# Creating functions

# Logit condicional 

def LL(beta):
    #Unpacking
    alpha1, alpha2, alpha3, betap, betaf=beta 
    
    #Utilidad representativa
    v1=alpha1+betap*df['price1']+betaf*df['feat1']
    v2=alpha2+betap*df['price2']+betaf*df['feat2']
    v3=alpha3+betap*df['price3']+betaf*df['feat3']
    v4=betap*df['price4']+betaf*df['feat4']
    
    #Exponencial de utilidades representativas
    expv1=v1.apply(np.exp)
    expv2=v2.apply(np.exp)
    expv3=v3.apply(np.exp)
    expv4=v4.apply(np.exp)
    
    #Inclusive value (denominador)
    inclvalue=expv1+expv2+expv3+expv4
    
    #Choice probabilities
    prob1=expv1/inclvalue
    prob2=expv2/inclvalue
    prob3=expv3/inclvalue
    prob4=expv4/inclvalue
    
    #Lilkelihood per observation
    l=df['brand1']*np.log(prob1)+df['brand2']*np.log(prob2)+df['brand3']*np.log(prob3)+df['brand4']*np.log(prob4)
    
    #Sumar likelihood sobre todas las observaciones
    return -np.sum(l)


# Nested Logit

## Structure (3,1)

def LL_N_31a(beta):
    #Unpacking
    alpha1, alpha2, alpha3, betap, betaf,gamma1=beta 
    
    # Nested parameters
    lambda1 = np.exp(gamma1)/(np.exp(gamma1) + 1)
    lambda2 = 1
    
    
    #Representative utilities
    v1=(alpha1+betap*df['price1']+betaf*df['feat1'])/lambda1
    v2=(alpha2+betap*df['price2']+betaf*df['feat2'])/lambda1
    v3=(alpha3+betap*df['price3']+betaf*df['feat3'])/lambda1
    v4=(betap*df['price4']+betaf*df['feat4'])/lambda2
    

    #Exp representative utilities
    expv1=v1.apply(np.exp)
    expv2=v2.apply(np.exp)
    expv3=v3.apply(np.exp)
    expv4=v4.apply(np.exp)
    
    # Nested
    
    a=expv1+expv2+expv3
    b=expv4
    
    #Inclusive value (denominador)
    inclvalue=(a)**(lambda1)+(b)**(lambda2)
    
    #Choice probabilities
    prob1=(expv1*(a)**(lambda1-1))/inclvalue
    prob2=(expv2*(a)**(lambda1-1))/inclvalue
    prob3=(expv3*(a)**(lambda1-1))/inclvalue
    prob4=(expv4*(b)**(lambda2-1))/inclvalue
    
    #Lilkelihood per observation
    l=df['brand1']*np.log(prob1)+df['brand2']*np.log(prob2)+df['brand3']*np.log(prob3)+df['brand4']*np.log(prob4)
    
    #Sumar likelihood sobre todas las observaciones
    return -np.sum(l)


def LL_N_31b(beta):
    #Unpacking
    alpha1, alpha2, alpha3, betap, betaf,gamma1=beta 
    
    # Nested parameters
    lambda1 = np.exp(gamma1)/(np.exp(gamma1) + 1)
    lambda2 = 1
    

    
    #Representative utilities
    v1=(alpha1+betap*df['price1']+betaf*df['feat1'])/lambda1
    v2=(alpha2+betap*df['price2']+betaf*df['feat2'])/lambda1
    v3=(alpha3+betap*df['price3']+betaf*df['feat3'])/lambda2
    v4=(betap*df['price4']+betaf*df['feat4'])/lambda1
    

    #Exp Representative utilities
    expv1=v1.apply(np.exp)
    expv2=v2.apply(np.exp)
    expv3=v3.apply(np.exp)
    expv4=v4.apply(np.exp)
    
    # Nested
    
    a=expv1+expv2+expv4
    b=expv3
    
    #Inclusive value (denominador)
    inclvalue=(a)**(lambda1)+(b)**(lambda2)
    
    #Choice probabilities
    prob1=(expv1*(a)**(lambda1-1))/inclvalue
    prob2=(expv2*(a)**(lambda1-1))/inclvalue
    prob3=(expv3*(b)**(lambda2-1))/inclvalue
    prob4=(expv4*(a)**(lambda1-1))/inclvalue
    
    #Lilkelihood per observation
    l=df['brand1']*np.log(prob1)+df['brand2']*np.log(prob2)+df['brand3']*np.log(prob3)+df['brand4']*np.log(prob4)
    
    #Sumar likelihood sobre todas las observaciones
    return -np.sum(l)

def LL_N_31c(beta):
    #Unpacking
    alpha1, alpha2, alpha3, betap, betaf,gamma1=beta 
    
    # Nested parameters
    lambda1 = np.exp(gamma1)/(np.exp(gamma1) + 1)
    lambda2 = 1
    
    
    #Representative utilities
    v1=(alpha1+betap*df['price1']+betaf*df['feat1'])/lambda1
    v2=(alpha2+betap*df['price2']+betaf*df['feat2'])/lambda2
    v3=(alpha3+betap*df['price3']+betaf*df['feat3'])/lambda1
    v4=(betap*df['price4']+betaf*df['feat4'])/lambda1
    

    #Exp Representative utilities
    expv1=v1.apply(np.exp)
    expv2=v2.apply(np.exp)
    expv3=v3.apply(np.exp)
    expv4=v4.apply(np.exp)
    
    # Nested
    
    a=expv1+expv4+expv3
    b=expv2
    
    #Inclusive value (denominador)
    inclvalue=(a)**(lambda1)+(b)**(lambda2)
    
    #Choice probabilities
    prob1=(expv1*(a)**(lambda1-1))/inclvalue
    prob2=(expv2*(b)**(lambda2-1))/inclvalue
    prob3=(expv3*(a)**(lambda1-1))/inclvalue
    prob4=(expv4*(a)**(lambda1-1))/inclvalue
    
    #Lilkelihood per observation
    l=df['brand1']*np.log(prob1)+df['brand2']*np.log(prob2)+df['brand3']*np.log(prob3)+df['brand4']*np.log(prob4)
    
    #Sumar likelihood sobre todas las observaciones
    return -np.sum(l)

def LL_N_31d(beta):
    #Unpacking
    alpha1, alpha2, alpha3, betap, betaf,gamma1=beta 
    
    # Nested parameters
    lambda1 = np.exp(gamma1)/(np.exp(gamma1) + 1)
    lambda2 = 1

    
    #Utilidad representativa
    v1=(alpha1+betap*df['price1']+betaf*df['feat1'])/lambda2
    v2=(alpha2+betap*df['price2']+betaf*df['feat2'])/lambda1
    v3=(alpha3+betap*df['price3']+betaf*df['feat3'])/lambda1
    v4=(betap*df['price4']+betaf*df['feat4'])/lambda1
    

    #Exp Representative utilities
    expv1=v1.apply(np.exp)
    expv2=v2.apply(np.exp)
    expv3=v3.apply(np.exp)
    expv4=v4.apply(np.exp)
    
    # Nested
    
    a=expv4+expv2+expv3
    b=expv1
    
    #Inclusive value (denominador)
    inclvalue=(a)**(lambda1)+(b)**(lambda2)
    
    #Choice probabilities
    prob1=(expv1*(b)**(lambda2-1))/inclvalue
    prob2=(expv2*(a)**(lambda1-1))/inclvalue
    prob3=(expv3*(a)**(lambda1-1))/inclvalue
    prob4=(expv4*(a)**(lambda1-1))/inclvalue
    
    #Lilkelihood per observation
    l=df['brand1']*np.log(prob1)+df['brand2']*np.log(prob2)+df['brand3']*np.log(prob3)+df['brand4']*np.log(prob4)
    
    #Sumar likelihood sobre todas las observaciones
    return -np.sum(l)

## Estructura (2,2)

def LL_N_22a(beta):
    #Unpacking
    alpha1, alpha2, alpha3, betap, betaf,gamma1,gamma2=beta 
    
    # Nested parameters
    lambda1 = np.exp(gamma1)/(np.exp(gamma1) + 1)
    lambda2 = np.exp(gamma2)/(np.exp(gamma2) + 1)
    
    #Representative utilities
    v1=(alpha1+betap*df['price1']+betaf*df['feat1'])/lambda1
    v2=(alpha2+betap*df['price2']+betaf*df['feat2'])/lambda1
    v3=(alpha3+betap*df['price3']+betaf*df['feat3'])/lambda2
    v4=(betap*df['price4']+betaf*df['feat4'])/lambda2
    

    #Exp Representative utilities
    expv1=v1.apply(np.exp)
    expv2=v2.apply(np.exp)
    expv3=v3.apply(np.exp)
    expv4=v4.apply(np.exp)
    
    # Nested
    
    a = expv1 + expv2
    b = expv3 + expv4
    
    #Inclusive value (denominador)
    inclvalue=(a)**(lambda1)+(b)**(lambda2)
    
    #Choice probabilities
    prob1=(expv1*(a)**(lambda1-1))/inclvalue
    prob2=(expv2*(a)**(lambda1-1))/inclvalue
    prob3=(expv3*(b)**(lambda2-1))/inclvalue
    prob4=(expv4*(b)**(lambda2-1))/inclvalue
    
    #Lilkelihood per observation
    l=df['brand1']*np.log(prob1)+df['brand2']*np.log(prob2)+df['brand3']*np.log(prob3)+df['brand4']*np.log(prob4)
    
    #Sumar likelihood sobre todas las observaciones
    return -np.sum(l)


def LL_N_22b(beta):
    #Unpacking
    alpha1, alpha2, alpha3, betap, betaf,gamma1,gamma2=beta 
    
    # Nested parameters
    lambda1 = np.exp(gamma1)/(np.exp(gamma1) + 1)
    lambda2 = np.exp(gamma2)/(np.exp(gamma2) + 1)

    
    #Representative utilities
    v1=(alpha1+betap*df['price1']+betaf*df['feat1'])/lambda1
    v2=(alpha2+betap*df['price2']+betaf*df['feat2'])/lambda2
    v3=(alpha3+betap*df['price3']+betaf*df['feat3'])/lambda1
    v4=(betap*df['price4']+betaf*df['feat4'])/lambda2
    

    #Exp Representative utilities
    expv1=v1.apply(np.exp)
    expv2=v2.apply(np.exp)
    expv3=v3.apply(np.exp)
    expv4=v4.apply(np.exp)
    
    # Nested
    
    a = expv1 + expv3 
    b = expv2 + expv4
    
    #Inclusive value (denominador)
    inclvalue=(a)**(lambda1)+(b)**(lambda2)
    
    #Choice probabilities
    prob1=(expv1*(a)**(lambda1-1))/inclvalue
    prob2=(expv2*(b)**(lambda2-1))/inclvalue
    prob3=(expv3*(a)**(lambda1-1))/inclvalue
    prob4=(expv4*(b)**(lambda2-1))/inclvalue
    
    #Lilkelihood per observation
    l=df['brand1']*np.log(prob1)+df['brand2']*np.log(prob2)+df['brand3']*np.log(prob3)+df['brand4']*np.log(prob4)
    
    #Sumar likelihood sobre todas las observaciones
    return -np.sum(l)

def LL_N_22c(beta):
    #Unpacking
    alpha1, alpha2, alpha3, betap, betaf,gamma1,gamma2=beta 
    
    # Nested parameters
    lambda1 = np.exp(gamma1)/(np.exp(gamma1) + 1)
    lambda2 = np.exp(gamma2)/(np.exp(gamma2) + 1)

    
    #Representative utilities
    v1=(alpha1+betap*df['price1']+betaf*df['feat1'])/lambda1
    v2=(alpha2+betap*df['price2']+betaf*df['feat2'])/lambda2
    v3=(alpha3+betap*df['price3']+betaf*df['feat3'])/lambda2
    v4=(betap*df['price4']+betaf*df['feat4'])/lambda1
    

    #Exp Representative utilities
    expv1=v1.apply(np.exp)
    expv2=v2.apply(np.exp)
    expv3=v3.apply(np.exp)
    expv4=v4.apply(np.exp)
    
    # Nested
    
    a = expv1 + expv4
    b = expv2 + expv3
    
    #Inclusive value (denominador)
    inclvalue=(a)**(lambda1)+(b)**(lambda2)
    
    #Choice probabilities
    prob1=(expv1*(a)**(lambda1-1))/inclvalue
    prob2=(expv2*(b)**(lambda2-1))/inclvalue
    prob3=(expv3*(b)**(lambda2-1))/inclvalue
    prob4=(expv4*(a)**(lambda1-1))/inclvalue
    
    #Lilkelihood per observation
    l=df['brand1']*np.log(prob1)+df['brand2']*np.log(prob2)+df['brand3']*np.log(prob3)+df['brand4']*np.log(prob4)
    
    #Sumar likelihood sobre todas las observaciones
    return -np.sum(l)


# Optimize with diferents methods

## Save values

def optimizar_logit_condicional_save (point0, metodo):
    conca = pd.DataFrame()
    
    ## Añadir beta óptimo obtenido
    
    a = minimize(LL,point0,method=metodo)
    
    a_values = a["x"]
        
    conca = pd.concat([conca,pd.DataFrame.transpose(pd.DataFrame(a_values))],axis=0)

    conca.rename({0: 'alph1',1: 'alph2',2: 'alph3',3: 'bp',4: 'bf'}, axis=1, inplace=True)
    conca.reset_index(drop=True, inplace=True)
    
    ## Añadir valor optimo obtenido
    val_a = -LL(a_values)
    
    val_opt = pd.DataFrame([val_a])
    val_opt.rename({0: 'valor_opt'}, axis=1, inplace=True)
    
    conca = pd.concat([conca,val_opt],axis=1)
    
    ## Añadir punto inicial empleado
    
    point_inicial = pd.DataFrame([point0])
    point_inicial.rename({0: 'alph1_0',1: 'alph2_0',2: 'alph3_0',3: 'bp_0',4: 'bf_0'}, axis=1, inplace=True)
    conca = pd.concat([conca,point_inicial],axis=1)
    
    ## Añadir nombre del metodo empleado
    
    name_method = pd.DataFrame([metodo])
    name_method.rename({0: 'Metodo'}, axis=1, inplace=True)
    conca = pd.concat([conca,name_method],axis=1)
    
    ## Añadir AIC (-2 * mv_betas) + (2*5) 
    
    AIC = pd.DataFrame([(2*5) -2*val_a])
    AIC.rename({0: 'AIC'}, axis=1, inplace=True)
    conca = pd.concat([conca,AIC],axis=1)
    
    ## Añadir Indice de razón de verosimilitud
    ceros = [0,0,0,0,0]
    valcero_a = -LL(ceros)
    
    razon_vero = pd.DataFrame([1-val_a/valcero_a])
    razon_vero.rename({0: 'Indice Verosimilitud'}, axis=1, inplace=True)
    conca = pd.concat([conca,razon_vero],axis=1)
    
    ## Añadir errores
    
    conca_SE = pd.DataFrame()
    
    ####Hessiano 
    HLL = nd.Hessian(LL)(a_values)
    ####Matriz de informacion ( no tomo -HLL porque estoy considerando que el modelo se multiplica por -1)
    HLL_inverse = np.linalg.inv(HLL)
    ####Extraer varianzas y calcular SE
    SE = pd.DataFrame(np.sqrt(np.diag(HLL_inverse)))
    conca_SE = pd.concat([conca_SE,SE],axis=1) 
    
    conca_SE = pd.DataFrame.transpose(conca_SE)
    
    conca_SE.rename({0: 'alph1_SE',1: 'alph2_SE',2: 'alph3_SE',3: 'bp_SE',4: 'bf_SE'}, axis=1, inplace=True)
    conca_SE.reset_index(drop=True, inplace=True)
    
    conca = pd.concat([conca,conca_SE],axis=1)
    
    # Añadir nidos
    
    nid = pd.DataFrame()
    n1 = [0]
    n2 = [0]
    n3 = [0]
    n4 = [0]

    nid['nest1'] = n1
    nid['nest2'] = n2
    nid['nest3'] = n3
    nid['nest4'] = n4
    nid["nest_est"] = "Logit condicional"
    
    nid.reset_index(drop=True, inplace=True)
    conca = pd.concat([conca,nid],axis=1)
    
    return conca


def optimizar_logit_31_save (point0, metodo):
 
    conca = pd.DataFrame()
    
    ## Añadir beta óptimo obtenido
    
    a = minimize(LL_N_31a,point0,method=metodo)
    b = minimize(LL_N_31b,point0,method=metodo)
    c = minimize(LL_N_31c,point0,method=metodo)
    d = minimize(LL_N_31d,point0,method=metodo)
    
    a_values,b_values,c_values,d_values = a["x"],b["x"],c["x"],d["x"]
    
    
    conca = pd.concat([conca,pd.DataFrame.transpose(pd.DataFrame(a_values))],axis=0)
    conca = pd.concat([conca,pd.DataFrame.transpose(pd.DataFrame(b_values))],axis=0)    
    conca = pd.concat([conca,pd.DataFrame.transpose(pd.DataFrame(c_values))],axis=0)
    conca = pd.concat([conca,pd.DataFrame.transpose(pd.DataFrame(d_values))],axis=0)
    
    conca.rename({0: 'alph1',1: 'alph2',2: 'alph3',3: 'bp',4: 'bf',5: "gamma1"}, axis=1, inplace=True)
    conca.reset_index(drop=True, inplace=True)
    
    ## Añadir valor optimo obtenido
    val_a, val_b, val_c, val_d = -LL_N_31a(a_values), -LL_N_31b(b_values), -LL_N_31c(c_values), -LL_N_31d(d_values)
    
    val_opt = pd.DataFrame([val_a, val_b, val_c, val_d])
    val_opt.rename({0: 'valor_opt'}, axis=1, inplace=True)
    
    conca = pd.concat([conca,val_opt],axis=1)
    
    ## Añadir punto inicial empleado
    
    point_inicial = pd.DataFrame([point0, point0, point0, point0])
    point_inicial.rename({0: 'alph1_0',1: 'alph2_0',2: 'alph3_0',3: 'bp_0',4: 'bf_0',5: "gamma1_0"}, axis=1, inplace=True)
    conca = pd.concat([conca,point_inicial],axis=1)
    
    ## Añadir nombre del metodo empleado
    
    name_method = pd.DataFrame([metodo,metodo, metodo, metodo])
    name_method.rename({0: 'Metodo'}, axis=1, inplace=True)
    conca = pd.concat([conca,name_method],axis=1)
    
    ## Añadir AIC (-2 * mv_betas) + (2*5) 
    
    AIC = pd.DataFrame([(2*6) -2*val_a, (2*6) -2*val_b, (2*6) -2*val_c, (2*6) -2*val_d])
    AIC.rename({0: 'AIC'}, axis=1, inplace=True)
    conca = pd.concat([conca,AIC],axis=1)
    
    ## Añadir Indice de razón de verosimilitud
    ceros = [0,0,0,0,0,0]
    valcero_a, valcero_b, valcero_c, valcero_d = -LL_N_31a(ceros), -LL_N_31b(ceros), -LL_N_31c(ceros), -LL_N_31d(ceros)
    
    razon_vero = pd.DataFrame([1-val_a/valcero_a, 1-val_b/valcero_b, 1-val_c/valcero_c, 1-val_d/valcero_d])
    razon_vero.rename({0: 'Indice Verosimilitud'}, axis=1, inplace=True)
    conca = pd.concat([conca,razon_vero],axis=1)
    
    ## Añadir errores
    
    conca_SE = pd.DataFrame()
    
    ### 31a
    ####Hessiano 
    HLL_31a = nd.Hessian(LL_N_31a)(a_values)
    ####Matriz de informacion ( no tomo -HLL porque estoy considerando que el modelo se multiplica por -1)
    HLL_inverse_31a = np.linalg.inv(HLL_31a)
    ####Extraer varianzas y calcular SE
    SE_31a = pd.DataFrame(np.sqrt(np.diag(HLL_inverse_31a)))
    conca_SE = pd.concat([conca_SE,SE_31a],axis=1)
    
    ### 31b
    ####Hessiano 
    HLL_31b = nd.Hessian(LL_N_31b)(b_values)

    ####Matriz de informacion ( no tomo -HLL porque estoy considerando que el modelo se multiplica por -1)
    HLL_inverse_31b = np.linalg.inv(HLL_31b)
    ####Extraer varianzas y calcular SE
    SE_31b = pd.DataFrame(np.sqrt(np.diag(HLL_inverse_31b)))
    conca_SE = pd.concat([conca_SE,SE_31b],axis=1)

    ### 31c
    ####Hessiano 
    HLL_31c = nd.Hessian(LL_N_31c)(c_values)
    ####Matriz de informacion ( no tomo -HLL porque estoy considerando que el modelo se multiplica por -1)
    HLL_inverse_31c = np.linalg.inv(HLL_31c)
    ####Extraer varianzas y calcular SE
    SE_31c = pd.DataFrame(np.sqrt(np.diag(HLL_inverse_31c)))
    conca_SE = pd.concat([conca_SE,SE_31c],axis=1)
    
    ### 31d
    ####Hessiano 
    HLL_31d = nd.Hessian(LL_N_31d)(d_values)
    ####Matriz de informacion ( no tomo -HLL porque estoy considerando que el modelo se multiplica por -1)
    HLL_inverse_31d = np.linalg.inv(HLL_31d)
    ####Extraer varianzas y calcular SE
    SE_31d = pd.DataFrame(np.sqrt(np.diag(HLL_inverse_31d)))
    conca_SE = pd.concat([conca_SE,SE_31d],axis=1)
    
    conca_SE = pd.DataFrame.transpose(conca_SE)
    
    conca_SE.rename({0: 'alph1_SE',1: 'alph2_SE',2: 'alph3_SE',3: 'bp_SE',4: 'bf_SE',5: "gamma1_SE"}, axis=1, inplace=True)
    conca_SE.reset_index(drop=True, inplace=True)
    
    conca = pd.concat([conca,conca_SE],axis=1)
    
    
    ## Añadir errores estándar de lambdas con método delta
    
    ### 31a    
    estim31a = (a["x"])
    def f_ee_31a(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [derivada]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_31a)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l31a = f_ee_31a(estim31a[5])
    ee_l31a = ee_l31a[0]
    
    
    ### 31b   
    estim31b = (b["x"])
    def f_ee_31b(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [derivada]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_31b)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l31b = f_ee_31b(estim31b[5])
    ee_l31b = ee_l31b[0]
    
    ### 31c    
    estim31c = (c["x"])
    def f_ee_31c(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [derivada]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_31c)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l31c = f_ee_31c(estim31c[5])
    ee_l31c = ee_l31c[0]
    
    ### 31d   
    estim31d = (d["x"])
    def f_ee_31d(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [derivada]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_31d)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l31d = f_ee_31d(estim31d[5])
    ee_l31d = ee_l31d[0]
    
    
    val_ee_lambda = pd.DataFrame([ee_l31a, ee_l31b, ee_l31c, ee_l31d])
    val_ee_lambda.rename({0: 'ee_lambda1'}, axis=1, inplace=True)
     
    conca = pd.concat([conca,val_ee_lambda],axis=1)
    
    
    ## Añadir nidos
    
    nid = pd.DataFrame()
    n1 = [1,1,1,2]
    n2 = [1,1,2,1]
    n3 = [1,2,1,1]
    n4 = [2,1,1,1]

    nid['nest1'] = n1
    nid['nest2'] = n2
    nid['nest3'] = n3
    nid['nest4'] = n4
    nid["nest_est"] = "(3,1)"
    
    ## Añadir lambdas
    nid["lambda1"] = np.exp(conca["gamma1"])/(np.exp(conca["gamma1"]) + 1)
    nid["lambda2"] = 1
    
    nid.reset_index(drop=True, inplace=True)
    conca = pd.concat([conca,nid],axis=1)
    
    return conca


def optimizar_logit_22_save (point0, metodo):
 
    conca = pd.DataFrame()
    
    ## Añadir beta óptimo obtenido
    
    a = minimize(LL_N_22a,point0,method=metodo)
    b = minimize(LL_N_22b,point0,method=metodo)
    c = minimize(LL_N_22c,point0,method=metodo)
    
    a_values,b_values,c_values = a["x"],b["x"],c["x"]
    
    
    conca = pd.concat([conca,pd.DataFrame.transpose(pd.DataFrame(a_values))],axis=0)
    conca = pd.concat([conca,pd.DataFrame.transpose(pd.DataFrame(b_values))],axis=0)    
    conca = pd.concat([conca,pd.DataFrame.transpose(pd.DataFrame(c_values))],axis=0)
    
    conca.rename({0: 'alph1',1: 'alph2',2: 'alph3',3: 'bp',4: 'bf',5: "gamma1",6: "gamma2"}, axis=1, inplace=True)
    conca.reset_index(drop=True, inplace=True)
    
    ## Añadir valor optimo obtenido
    val_a, val_b, val_c = -LL_N_22a(a_values), -LL_N_22b(b_values), -LL_N_22c(c_values)
    
    val_opt = pd.DataFrame([val_a, val_b, val_c])
    val_opt.rename({0: 'valor_opt'}, axis=1, inplace=True)
    
    conca = pd.concat([conca,val_opt],axis=1)
    
    ## Añadir punto inicial empleado
    
    point_inicial = pd.DataFrame([point0, point0, point0])
    point_inicial.rename({0: 'alph1_0',1: 'alph2_0',2: 'alph3_0',3: 'bp_0',4: 'bf_0',5: "gamma1_0",6: "gamma2_0"}, axis=1, inplace=True)
    conca = pd.concat([conca,point_inicial],axis=1)
    
    ## Añadir nombre del metodo empleado
    
    name_method = pd.DataFrame([metodo,metodo, metodo])
    name_method.rename({0: 'Metodo'}, axis=1, inplace=True)
    conca = pd.concat([conca,name_method],axis=1)
    
    ## Añadir AIC (-2 * mv_betas) + (2*5) 
    
    AIC = pd.DataFrame([(2*7) -2*val_a, (2*7) -2*val_b, (2*7) -2*val_c])
    AIC.rename({0: 'AIC'}, axis=1, inplace=True)
    conca = pd.concat([conca,AIC],axis=1)
    
    ## Añadir Indice de razón de verosimilitud
    ceros = [0,0,0,0,0,0,0]
    valcero_a, valcero_b, valcero_c= -LL_N_22a(ceros), -LL_N_22b(ceros), -LL_N_22c(ceros)
    
    razon_vero = pd.DataFrame([1-val_a/valcero_a, 1-val_b/valcero_b, 1-val_c/valcero_c])
    razon_vero.rename({0: 'Indice Verosimilitud'}, axis=1, inplace=True)
    conca = pd.concat([conca,razon_vero],axis=1)
    
    ## Añadir errores
    
    conca_SE = pd.DataFrame()
    
    ### 22a
    ####Hessiano 
    HLL_22a = nd.Hessian(LL_N_22a)(a_values)
    ####Matriz de informacion ( no tomo -HLL porque estoy considerando que el modelo se multiplica por -1)
    HLL_inverse_22a = np.linalg.inv(HLL_22a)
    ####Extraer varianzas y calcular SE
    SE_22a = pd.DataFrame(np.sqrt(np.diag(HLL_inverse_22a)))
    conca_SE = pd.concat([conca_SE,SE_22a],axis=1)
    
    ### 22b
    ####Hessiano 
    HLL_22b = nd.Hessian(LL_N_22b)(b_values)

    ####Matriz de informacion ( no tomo -HLL porque estoy considerando que el modelo se multiplica por -1)
    HLL_inverse_22b = np.linalg.inv(HLL_22b)
    ####Extraer varianzas y calcular SE
    SE_22b = pd.DataFrame(np.sqrt(np.diag(HLL_inverse_22b)))
    conca_SE = pd.concat([conca_SE,SE_22b],axis=1)

    ### 22c
    ####Hessiano 
    HLL_22c = nd.Hessian(LL_N_22c)(c_values)
    ####Matriz de informacion ( no tomo -HLL porque estoy considerando que el modelo se multiplica por -1)
    HLL_inverse_22c = np.linalg.inv(HLL_22c)
    ####Extraer varianzas y calcular SE
    SE_22c = pd.DataFrame(np.sqrt(np.diag(HLL_inverse_22c)))
    conca_SE = pd.concat([conca_SE,SE_22c],axis=1)
      
    conca_SE = pd.DataFrame.transpose(conca_SE)
    
    conca_SE.rename({0: 'alph1_SE',1: 'alph2_SE',2: 'alph3_SE',3: 'bp_SE',4: 'bf_SE',5: "gamma1_SE",6: "gamma2_SE"}, axis=1, inplace=True)
    conca_SE.reset_index(drop=True, inplace=True)
    
    conca = pd.concat([conca,conca_SE],axis=1)
    
    ## Añadir errores estándar de lambdas con método delta
    
    ### 22a    
    estim22a = (a["x"])
    def f_ee_22a(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [derivada], [0]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_22a)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l1_22a = f_ee_22a(estim22a[5])
    ee_l1_22a = ee_l1_22a[0]
    
    def f_ee_22a2(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [0] , [derivada]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_22a)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l2_22a = f_ee_22a2(estim22a[6])
    ee_l2_22a = ee_l2_22a[0]
    
    ### 22b    
    estim22b = (b["x"])
    def f_ee_22b(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [derivada], [0]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_22b)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l1_22b = f_ee_22b(estim22b[5])
    ee_l1_22b = ee_l1_22b[0]
    
    def f_ee_22b2(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [0] , [derivada]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_22b)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l2_22b = f_ee_22b2(estim22b[6])
    ee_l2_22b = ee_l2_22b[0]
    
    ### 22c    
    estim22c = (c["x"])
    def f_ee_22c(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [derivada], [0]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_22c)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l1_22c = f_ee_22c(estim22c[5])
    ee_l1_22c = ee_l1_22c[0]
    
    def f_ee_22c2(g):
        derivada = np.exp(g)/((1+ np.exp(g))**2)
        mat = [[0], [0], [0], [0], [0], [0] , [derivada]]
        mat_t = np.transpose(mat)
        res = np.dot(mat_t, HLL_inverse_22c)
        res2 = np.dot(res, mat)
        ee = np.sqrt(res2)
        return ee 
    
    ee_l2_22c = f_ee_22c2(estim22c[6])
    ee_l2_22c = ee_l2_22c[0]    
    
    
    val_ee_lambda1 = pd.DataFrame([ee_l1_22a, ee_l1_22b, ee_l1_22c])
    val_ee_lambda1.rename({0: 'ee_lambda1'}, axis=1, inplace=True)
     
    conca = pd.concat([conca,val_ee_lambda1],axis=1)
    
    val_ee_lambda2 = pd.DataFrame([ee_l2_22a, ee_l2_22b, ee_l2_22c])
    val_ee_lambda2.rename({0: 'ee_lambda2'}, axis=1, inplace=True)
     
    conca = pd.concat([conca,val_ee_lambda2],axis=1)
    
    # Añadir nidos
    
    nid = pd.DataFrame()
    n1 = [1,1,1]
    n2 = [1,2,2]
    n3 = [2,1,2]
    n4 = [2,2,1]

    
    nid['nest1'] = n1
    nid['nest2'] = n2
    nid['nest3'] = n3
    nid['nest4'] = n4
    nid["nest_est"] = "(2,2)"
    
    ## Añadir lambdas
    nid["lambda1"] = np.exp(conca["gamma1"])/(np.exp(conca["gamma1"]) + 1)
    nid["lambda2"] = np.exp(conca["gamma2"])/(np.exp(conca["gamma2"]) + 1)
    
    nid.reset_index(drop=True, inplace=True)
    conca = pd.concat([conca,nid],axis=1)
   
    return conca

# Creacion de la base de datos

##Correr los modelos con distintos metodos y puntos iniciales

### Logit condicional


base_giant_1 = pd.DataFrame()

try:    
    save_val = optimizar_logit_condicional_save([0,0,0,0,0],'trust-constr')
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_condicional_save([1.38,0.83,-1.65,-26.58,0.37],'trust-constr')
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_condicional_save([0,0,0,0,0],"L-BFGS-B")
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_condicional_save([1.38,0.83,-1.65,-26.58,0.37],"L-BFGS-B")
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_condicional_save([0,0,0,0,0],"Nelder-Mead")
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_condicional_save([1.38,0.83,-1.65,-26.58,0.37],"Nelder-Mead")
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_condicional_save([0,0,0,0,0],"Powell")
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_condicional_save([1.38,0.83,-1.65,-26.58,0.37],"Powell")
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_condicional_save([0,0,0,0,0],"TNC")
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_condicional_save([1.38,0.83,-1.65,-26.58,0.37],"TNC")
    base_giant_1 = pd.concat([base_giant_1,save_val],axis=0)
    base_giant_1.reset_index(drop=True, inplace=True)
except:
    pass


### Nested logit

base_giant_31 = pd.DataFrame()
try:    
    save_val = optimizar_logit_31_save([0,0,0,0,0,0],'trust-constr')
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_31_save([1.38,0.83,-1.65,-26.58,0.37,0.59],'trust-constr')
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_31_save([0,0,0,0,0,0],"L-BFGS-B")
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_31_save([1.38,0.83,-1.65,-26.58,0.37,0.59],"L-BFGS-B")
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_31_save([0,0,0,0,0,0],"Nelder-Mead")
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_31_save([1.38,0.83,-1.65,-26.58,0.37,0.59],"Nelder-Mead")
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_31_save([0,0,0,0,0,0],"Powell")
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass


try:
    save_val = optimizar_logit_31_save([1.38,0.83,-1.65,-26.58,0.37,0.59],"Powell")
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_31_save([0,0,0,0,0,0],"TNC")
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass


try:
    save_val = optimizar_logit_31_save([1.38,0.83,-1.65,-26.58,0.37,0.59],"TNC")
    base_giant_31 = pd.concat([base_giant_31,save_val],axis=0)
    base_giant_31.reset_index(drop=True, inplace=True)
except:
    pass




base_giant_22 = pd.DataFrame()

try:
    save_val = optimizar_logit_22_save([0,0,0,0,0,0,0],'trust-constr')
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass

try: 
    save_val = optimizar_logit_22_save([1.38,0.83,-1.65,-26.58,0.37,0.59,1],'trust-constr')
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_22_save([0,0,0,0,0,0,0],"L-BFGS-B")
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass


try:
    save_val = optimizar_logit_22_save([1.38,0.83,-1.65,-26.58,0.37,0.59,1],"L-BFGS-B")
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_22_save([0,0,0,0,0,0,0],"Nelder-Mead")
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_22_save([1.38,0.83,-1.65,-26.58,0.37,0.59,1],"Nelder-Mead")
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_22_save([0,0,0,0,0,0,0],"Powell")
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass


try:
    save_val = optimizar_logit_22_save([1.38,0.83,-1.65,-26.58,0.37,0.59,1],"Powell")
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass

try:
    save_val = optimizar_logit_22_save([0,0,0,0,0,0,0],"TNC")
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass


try:
    save_val = optimizar_logit_22_save([1.38,0.83,-1.65,-26.58,0.37,0.59,1],"TNC")
    base_giant_22 = pd.concat([base_giant_22,save_val],axis=0)
    base_giant_22.reset_index(drop=True, inplace=True)
except:
    pass

##Eliminar el archivo csv en caso de que exista

try:
    os.remove('logit_condicional.csv')  
except:
    pass

try:
    os.remove('Nested_logit_31.csv')  
except:
    pass

try:
    os.remove('Nested_logit_22.csv')       
except:
    pass 

try:
    os.remove('Nested_logit_all.csv')   
except:
    pass

#Guardar los modelos en archivo csv

base_giant_22.to_csv('Nested_logit_22.csv', mode='a', index=False, header=True) 
base_giant_22 = base_giant_22.dropna()

base_giant_31.to_csv('Nested_logit_31.csv', mode='a', index=False, header=True) 
base_giant_31 = base_giant_31.dropna()

base_giant_1.to_csv('logit_condicional.csv', mode='a', index=False, header=True) 
base_giant_1 = base_giant_1.dropna()


base_giant_nested = pd.concat([base_giant_31,base_giant_22, base_giant_1],axis=0)


base_giant_nested = base_giant_nested.reindex(columns=\
                                              ['alph1','alph2','alph3',\
                                               'bp','bf','gamma1','gamma2',\
                                               "lambda1","lambda2",\
                                               'valor_opt','Metodo',\
                                               'AIC','Indice Verosimilitud',\
                                               "nest_est",\
                                               'nest1','nest2','nest3',\
                                               'nest4','alph1_0','alph2_0',\
                                               'alph3_0','bp_0','bf_0',\
                                               'gamma1_0','gamma2_0','alph1_SE',\
                                               'alph2_SE','alph3_SE','bp_SE',\
                                               'bf_SE','gamma1_SE','gamma2_SE',\
                                               'ee_lambda1','ee_lambda2'])  

    
base_giant_nested.to_csv('Nested_logit_all.csv', mode='a', index=False, header=True) 

#Combinación de nidos con el menor AIC
AIC_low = base_giant_nested[base_giant_nested.AIC == base_giant_nested.AIC.min()]
print(AIC_low)


