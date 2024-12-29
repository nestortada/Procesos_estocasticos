'''
Model: MDP for trading optimal policy
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

Estados = ['B','S']

def final_states(D=data, currency="JPY", n_per=30, only_states=True, rolling_std=False):  # la ventana de tiempo para la desviación móvil es 20 días
    currency = currency.upper()
    D = pd.DataFrame(data[currency])
    D['Mob_std'] = D[currency].rolling(n_per).std()
    D['Cambio_%'] = D[currency].pct_change()
    if rolling_std:
        D['sigma'] = D['Cambio_%'].rolling(n_per).std()
    D = D[n_per:]
    Estados = ['B','S']
    D['Estado'] = np.where(D['Cambio_%'] < 0, Estados[0], Estados[1])
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
    resultado = ((prob_obs - prob_esp) ** 2 / prob_esp).sum(axis=1) * freq_obs["Total"]
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

################################## Condiciones de modelo ###################################################

divisas = ['JPY','CHF'] # Cuales divisas se va a utilizar en el aprendizaje supervisado
comission = 0.01 # Es la comisión de cambiar de divisa
matrices = {divisas[i]:markoviana(divisas[i],prints=False) for i in range(len(divisas))} # Es un diccionario el cual contiene la matriz de trasición para cada divisa
sigma_1 = np.round(final_states(currency=divisas[0], only_states=True, rolling_std=True)['sigma'].tolist()[-1], 4) # Es la ultima desviación movil de la primera divisa
sigma_2 = np.round(final_states(currency=divisas[1], only_states=False, rolling_std=True)['sigma'].tolist()[-1], 4) # Es la ultima desviación movil de la segunda divisa
matrix = matrices[divisas[0]] # Es la matriz de transición de la primera divisa
p_matrix = pd.DataFrame(columns=matrix.stack().index, index=matrix.stack().index) # Son todos los posibles estados de conjuntos de S y B
for i in p_matrix.index:
    for j in p_matrix.columns:
        p_matrix.loc[i,j] = ((matrices[divisas[0]].loc[i[0],j[0]]*matrices[divisas[1]].loc[i[1],j[1]]).round(4)) # Es la multiplicación de componente por componente para las dos divisas, genera un resultado de una matriz de transición de 4*4 que los estadosson duplas

################################## Funciones de modelo ###################################################

def Starting_State():  
    s1 = 1 # Es el inicio del día antes que se abra el mercado 
    s2 = divisas[0] # Es la divisa en el primer dia que comienza
    s3 = 'Sube' # Es el estado que comienza en el primer dia para la primera divisa
    s4 = 'Baja' # Es el estado que comienza en el primer día para la segunda divisa
    s5 = 100 # Es la contidad incial que comienza en el primer dia con la primera divisa
    s6 = 100 # Es la cantidad incial que comienza en el primer día con la primera divisa
    s = (s1,s2,s3,s4,s5,s6)
    return s

def Action_Set(s):   
    return ['mantener divisa','cambiar divisa'] # Son las posibles acciones que puede tomar con los diferentes Estados 

def Event_Set(s,a):
    event_list = [[f'Sube {i}', f'Baja {i}'] for i in divisas] # Son las posibles combinaciones entre S y B teniendo en cuenta su divisa
    ES = list(itertools.product(*event_list)) # Es para que esas combinaciones se creen en una lista
    return ES    

def Transition_Equations(s,a,e):
    (s1, s2, s3, s4, s5, s6) = s
    sn1 = s1 + 1 # Es para que en el siguiente estado el dia avance una unidad
    if a == 'cambiar divisa':
        if s2 == divisas[0]:
            sn2 = divisas[1] 
        else:
            sn2 = divisas[0]
    elif a == 'mantener divisa':
        sn2 = s2 # Esto verifica si la acción es cambiar, si se cambia la moneda, se verifica que divisa estamos y sí se cambia a la otra. Sin embargo, si la acción es matener el siguiente estado para s2 sera en la moneda que este en este momento 

    if e == (f'Sube {divisas[0]}', f'Sube {divisas[1]}') or e == (f'Sube {divisas[0]}', f'Baja {divisas[1]}'):
        sn3 = 'Sube'
        sn5 = s5 * (1 + sigma_1)
    elif e == (f'Baja {divisas[0]}', f'Sube {divisas[1]}') or e == (f'Baja {divisas[0]}', f'Baja {divisas[1]}'):
        sn3 = 'Baja'
        sn5 = s5 * (1 - sigma_1)

    if e == (f'Baja {divisas[0]}', f'Sube {divisas[1]}') or e == (f'Sube {divisas[0]}', f'Sube {divisas[1]}'):
        sn4 = 'Sube'
        sn6 = s6 * (1 + sigma_2)
    elif e == (f'Baja {divisas[0]}', f'Baja {divisas[1]}') or e == (f'Sube {divisas[0]}', f'Baja {divisas[1]}'):
        sn4 = 'Baja'
        sn6 = s6 * (1 - sigma_2) # Verifica si el evento en el que esta en este momento  va a subir o va a bajar si para el evento es de bajada para S3 y S4 su siguiente estado va se B (Bajada), y s5 y s6 van a ser restado segun su desviación respectiva. En cambio si el evento siguiente es Sube S3 y S4 sus estados siguientes van a ser S (Subida), y S5 y s6 su precio va a aumentar 

    sn = (sn1, sn2, sn3, sn4, np.round(sn5,4), np.round(sn6,4))
    return sn

def Constraints(s,a,sn,L):
    (s1, s2, s3, s4, s5, s6) = s
    Ct = (s1 <= 3) # Aqui se tiene como restricción que el numero de dia al inicio del mercado no puede ser mayor a 3
    return Ct

def Transition_Probabilities(s,a,e):
    (s1, s2, s3, s4, s5, s6) = s
    return p_matrix.loc[(s3[0],s4[0]),tuple([event[0] for event in e])]
    
def Action_Contribution(s,a):
    (s1, s2, s3, s4, s5, s6) = s
    if a == 'cambiar divisa':
        if s2 == divisas[0]:
            ca = -comission * s5
        else:
            ca = -comission * s6
    else:
        ca = 0 # Se verifia si acción fue cambiar si la acción fue cambiar se verifica en que divisa estoy en este momento y se cobrara una comisión del 0.01 del precio de la divisa en ese momento. Si la acción es mantener no se cobrara ninguna comisión 
    return np.round(ca, 4)

def Event_Contribution(s,a,e):
    (s1, s2, s3, s4, s5, s6) = s
    if s2 == divisas[0]:
        if e == (f'Sube {divisas[0]}', f'Sube {divisas[1]}') or e == (f'Sube {divisas[0]}', f'Baja {divisas[1]}'):
            ce = s5 * sigma_1
        else:
            ce = -s5 * sigma_1
    else:
        if e == (f'Sube {divisas[0]}', f'Sube {divisas[1]}') or e == (f'Baja {divisas[0]}', f'Sube {divisas[1]}'):
            ce = s6 * sigma_2
        else:
            ce = -s6 * sigma_2 # La contribución del evente verifica cual es el evento y se suma o se resta el precio de la divisa por el sigma correspondiente dependiendo si sube o baja la divisa 
    return np.round(ce, 4)

def Quality_Function(m,p,ca,ce,V_sn):
    Q_s_a = ca + sum(p[i] * (ce[i] + V_sn[i]) for i in range(0, m))
    return np.round(Q_s_a, 4) # La función de la calidad tiene todos los estados S y todos las acciones A y calcula la utilidad dependiendo de la contribución de la acción, del evente , la probabilidad y su estado futuro 

def Optimal_Value_Function(Q_s_a):
    V_s = max(Q_s_a) # Con todas las acciones y estados se maxima para determinar cual es la mejor acción que se debe tomar para cada estado
    return V_s

def Boundary_Condition(s):    
    (s1, s2, s3, s4, s5, s6) = s
    if s1 == 4:
        if s2 == divisas[0]:
            V_s = np.round(s5, 4)
        else:
            V_s = np.round(s6, 4)
    else:
        V_s = 0 # Se verifica si se s1 llega al inicio del 4 dia antes que se abra los mercados, si se llega el dia todos los estados s1 4 su condición de contorno este determinado por el precio de la divisa ese dia. Si no se ha llegado al cuerto dia su condición de contorno sera de 0 
    return V_s
