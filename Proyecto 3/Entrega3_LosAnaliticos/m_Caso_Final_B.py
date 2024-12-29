'''
Model: All or nothing (2 state variables)
Objective: Profit Maximization

'''

import pandas as pd #Para manejo de datos en DataFrames y funciónes de analítica de datos
import numpy as np #Para manejo de datos numéricos y arreglos
import matplotlib.pyplot as plt #Para la visualización de datos
import seaborn as sns #Para la visualización de datos
import scipy.stats as stats #Para realizar pruebas de hipótesis y estadiísticas
import warnings #Para evitar promptes de advertencia
warnings.filterwarnings('ignore') # evita warnings innecesarios al ejecutar código
import itertools


def load_data(url):
    df = pd.read_csv(url)
    df.columns = ['Fecha', ' Año', 'Nombre_Moneda', 'Continente', 'Cambio', 'Tipo_tasa', 'TRM', 'Id_Moneda']
    weekends = df['Fecha'].unique().tolist()[2::7]+df['Fecha'].unique().tolist()[3::7]
    keep = set(df['Fecha'].unique().tolist())
    dates = keep.difference(weekends)
    dates =list(dates)
    dates.sort()
    df = df[df['Fecha'].isin(dates)]
    COP_Med_Data = df.query("Cambio == 'Pesos colombianos por cada moneda' and Tipo_tasa == 'Media'")
    COP_Med_Data['TRM'] = COP_Med_Data['TRM'].str.replace(',','.').astype(float)
    data_original = pd.DataFrame(columns=df['Id_Moneda'].unique().tolist(), index=df['Fecha'].unique())
    for i in data_original.index:
        row = list(COP_Med_Data.query("Fecha == @i")['TRM'])
        data_original.loc[i] = row
    data_original = data_original.astype(float)
    data_original.head()
    data = data_original.copy()
    data.reset_index(drop=True, inplace=True)
    return data

link_database = 'https://raw.githubusercontent.com/NicolasCacer/bases-de-datos/main/1.2.TCM_Serie%20para%20un%20rango%20de%20fechas%20dado.csv'

data = load_data(url=link_database)

Estados = ['B3','B2','B1','S1','S2','S3']
def final_states(D=data, currency="JPY", n_per=20,only_states=True, rolling_std = False):  # la ventana de tiempo para la desviación móvil es 20 días
    currency = currency.upper()
    D = pd.DataFrame(data[currency])
    D['Mob_std'] = D[currency].rolling(n_per).std()
    D['Cambio_%'] = D[currency].pct_change()
    if rolling_std:
        D['sigma'] = D['Cambio_%'].rolling(n_per).std()
    D = D[n_per:]
    D['Estado'] = np.where(D['Cambio_%'] < -2 * D['Mob_std'] / D[currency], Estados[0],
                    np.where(((-2 * D['Mob_std'] / D[currency] <= D['Cambio_%']) & (D['Cambio_%'] < -1 * D['Mob_std'] / D[currency])), Estados[1],
                    np.where(((-1 * D['Mob_std'] / D[currency] <= D['Cambio_%']) & (D['Cambio_%'] < 0)), Estados[2],
                    np.where(((0 <= D['Cambio_%']) & (D['Cambio_%'] < D['Mob_std'] / D[currency])), Estados[3],
                    np.where(((D['Mob_std'] / D[currency] <= D['Cambio_%']) & (D['Cambio_%'] < 2 * D['Mob_std'] / D[currency])), Estados[4],Estados[5])))))
    if only_states:
        D = D.drop(columns=[currency, 'Mob_std', 'Cambio_%'])
    D.reset_index(drop=True, inplace=True)
    return D

def markoviana(currency = "JPY", n_periods = 20, prints=True):

    Data = final_states(currency= currency, n_per= n_periods)
    Data.columns = ["x_t"]
    Data["x_t+1"] = Data["x_t"].shift(-1)
    Data["x_t+2"] = Data["x_t+1"].shift(-1)
    freq_obs = pd.DataFrame(columns=("Anterior", "Actual"))
    for i in Estados:
        for j in Estados:
            freq_obs.loc[len(freq_obs)] = [i,j]
    freq_obs = pd.concat([freq_obs,pd.DataFrame(np.zeros((freq_obs.shape[0],len(Estados)+1)),columns=Estados+['Total'])], axis=1)
    for i in range(len(Data)-2):
        secuencia = Data.loc[i].tolist()
        freq_obs.loc[(freq_obs['Anterior'] == secuencia[0]) & (freq_obs['Actual'] == secuencia[1]),secuencia[2]] += 1
    freq_obs['Total'] = freq_obs[Estados].sum(axis=1)
    freq_esp = freq_obs.copy()
    freq_esp.iloc[:,2:] = 0
    for i in range(len(Data)-2):
        secuencia = Data.loc[i].tolist()
        freq_esp.loc[freq_esp['Actual'] == secuencia[1],secuencia[2]] += 1
    freq_esp['Total'] = freq_esp[Estados].sum(axis=1)
    porcentaje_menor_5 = (freq_obs[Estados] < 5).mean() * 100
    hay_ceros = (freq_obs["Total"] == 0).any()
    porcentaje_menor_5 = (freq_obs[Estados] < 5).mean() * 100
    hay_ceros = (freq_obs["Total"] == 0).any()
    if (porcentaje_menor_5.mean() > 25 or hay_ceros) and prints:
        print("Se necesito agrupar dado que los datos menores a 5 represetan más del 25% del total o hay 0 en la columna total")
        print("---"*100)
        freq_obs = freq_obs.groupby("Anterior").sum().reset_index()
        freq_esp = freq_esp.groupby("Anterior").sum().reset_index()
    prob_obs = freq_obs[Estados].div(freq_obs["Total"], axis=0)
    prob_esp = freq_esp[Estados].div(freq_esp["Total"], axis=0)
    chi_2 = (((prob_obs - prob_esp) ** 2 / prob_esp).sum(axis=1) * freq_obs["Total"]).sum()
    deg_free = (prob_esp.shape[1]-1)*(prob_esp.shape[0]-1)
    alpha = 0.05
    chi2_inv = stats.chi2.ppf(1-alpha, deg_free)
    if chi_2 < chi2_inv:
        Markov = "No se rechaza Ho"
    else:
        Markov = "Se rechaza Ho"
    freq_Markov = pd.DataFrame(0,columns= Estados+["Total"], index=Estados)
    for i in range(len(Data)-1):
        secuencia = Data.loc[i].tolist()[:2]
        freq_Markov.loc[secuencia[0],secuencia[1]]+= 1
    freq_Markov["Total"] = freq_Markov.sum(axis=1)
    
    prob_Markov = freq_Markov[Estados].div(freq_Markov["Total"], axis=0)
    

    if prints:
        print(f"-> Hipotesis markoviana\n {Markov}")
        print("-----"*10)
        print(f"-> Matriz de transición")
        print(prob_Markov.round(4))
        
    return prob_Markov



divisas = ['JPY','CHF']  
comission = 0.01
sigmas = {}
for divisa in divisas:
    sigma_value = np.round(final_states(currency=divisa, only_states=True, rolling_std=True)['sigma'].tolist()[-1], 4)
    sigmas[divisa] = sigma_value 

Estados = ['S1', 'S2', 'S3', 'B1', 'B2', 'B3']  
matrices = {divisa: markoviana(divisa, prints=False) for divisa in divisas}


resultado_kron = np.array([[1]])
for key, matriz in matrices.items():
    resultado_kron = np.kron(resultado_kron, matriz.values)

indices = pd.MultiIndex.from_product([matriz.index for matriz in matrices.values()],
                                     names=[f'Matriz{i+1}' for i in range(len(matrices))])
nombres_indices = ["".join(tupla) for tupla in indices]


markov_F = pd.DataFrame(resultado_kron , columns=indices, index=indices)
    

def Starting_State():

    s1 = 1  
    s2 = divisas[0]  
    divisa_states = {divisa: 'S1' for divisa in divisas}  
    s_vals = {divisa: 100 for divisa in divisas}  

    
    return s1,s2, divisa_states , s_vals



def Action_Set(s):
    return ['mantener divisa', 'cambiar divisa']

def Event_Set(s,a):
    return indices


def Transition_Equations(s, a, e):
    s1, s2, divisa_states , s_vals = s
    sn1 = s1 + 1  # Incrementar el paso

    sn_divisa_states = divisa_states.copy()
    sn_vals = s_vals.copy()


    for i in range(len(divisas)): # Asignar el estado futuro de la moneda segun cada evento
        for estados in Estados:
            posibles =  [tupla for tupla in  indices if tupla[i] == estados]
            if e in posibles:
                sn_divisa_states[divisas[i]] = e[i]
                signo = 1 if e[i][0] == 'S' else -1
                sn_vals[divisas[i]] = np.round(s_vals[divisas[i]]* (signo * (0.5 + 1 * (int(e[i][1]) - 1)) * sigmas[divisas[i]] + 1),4)

    
    if a == 'cambiar divisa':
        current_index = divisas.index(s2)
        sn2 = divisas[(current_index + 1) % len(divisas)]
    else:
        sn2 = s2




    return sn1 , sn2 ,sn_divisa_states , sn_vals

def Constraints(s,a,sn,L):
    s1, *resto = s
    Ct = (s1 <= 3)
    return Ct


def Transition_Probabilities(s, a, e):
    s1,s2 ,divisa_states , s_vals = s
    divisa_states_values = list(divisa_states.values())
    return markov_F.loc[(tuple(divisa_states_values)),e]

    

def Action_Contribution(s, a):
    s1,s2, divisa_states, s_vals= s
    
    if a == 'cambiar divisa':
        return -comission * s_vals[s2]
    return 0

def Event_Contribution(s, a, e):
    s1, s2, divisa_states, s_vals = s
 

    for i in range(len(divisas)): # Asignar el estado futuro de la moneda segun cada evento
        if s2 == divisas[i]:
            posibles =  [tupla for tupla in indices if tupla[i][0] == 'S']
            if e in posibles:
                ce = sigmas[divisas[i]]*s_vals[divisas[i]]
            else:
                ce = -sigmas[divisas[i]]*s_vals[divisas[i]]
    return np.round(ce,4)


def Quality_Function(m, p, ca, ce, V_sn):
    return np.round(ca + sum(p[i] * (ce[i] + V_sn[i]) for i in range(m)), 4)

def Optimal_Value_Function(Q_s_a):
    return  max(Q_s_a)

def Boundary_Condition(s):
    s1, s2, divisa_states, s_vals= s
    if s1 == 4:  # Paso final
        return np.round(s_vals[s2], 4)
    return 0
