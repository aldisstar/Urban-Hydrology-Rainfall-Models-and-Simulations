import pandas as pd

# Cargar datos
df = pd.read_csv('Rainfall_Daily_Data.csv', parse_dates=['Time'])

# Asegurarnos de que la columna de fecha esté ordenada
df = df.sort_values(by='Time')
df.set_index('Time', inplace=True)

#print(df.head())

# Definimos las duraciones en días para las cuales queremos calcular las precipitaciones acumuladas
durations = [1, 2, 3, 7, 14, 30]  

# Precipitaciones Máximas Acumuladas
max_precipitations = {station: {duration: [] for duration in durations} for station in df.columns}

for station in df.columns:
    for duration in durations:
        max_precipitations[station][duration] = df[station].rolling(window=duration).sum().dropna()

# Convertir el diccionario a un DataFrame para un análisis más fácil
max_precipitations_df = {station: pd.DataFrame(max_precipitations[station]).dropna() for station in df.columns}

# Imprimir las primeras filas de cada DataFrame
for station, data in max_precipitations_df.items():
    print(f"Estación {station}:")
    #print(data.head())

  
# Ajustar Distribuciones de Probabilidad    
from scipy.stats import genextreme as gev
import numpy as np

gev_params = {station: {} for station in df.columns}

for station in df.columns:
    for duration in durations:
        data = max_precipitations_df[station][duration].values
        if len(data) > 0:  # Asegurarse de que hay datos suficientes para ajustar
            params = gev.fit(data)
            gev_params[station][duration] = params

print(gev_params)

# Generar curvas DDF
import matplotlib.pyplot as plt

periodos_retorno = [2, 5, 10, 25, 50, 100]  # Periodos de retorno en años
probabilidades = 1 - 1 / np.array(periodos_retorno)

plt.figure(figsize=(10, 6))

for station in df.columns:
    plt.figure(figsize=(10, 6))
    for duration in durations:
        if duration in gev_params[station]:
            params = gev_params[station][duration]
            precip_values = gev.ppf(probabilidades, *params)
            plt.plot(periodos_retorno, precip_values, label=f'{duration} días')
    plt.xlabel('Periodo de retorno (años)')
    plt.ylabel('Precipitación acumulada (mm)')
    plt.title(f'Curvas DDF para la estación {station}')
    plt.legend()
    plt.grid(True)
    plt.show()







# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import gumbel_r

# # Leer los datos del archivo CSV
# df = pd.read_csv('Rainfall_Daily_Data.csv')

# # Crear una lista de duraciones y un diccionario para almacenar las precipitaciones máximas
# durations = [5, 10, 15, 30, 60, 120, 180, 360, 720, 1440]
# max_precipitations = {duration: [] for duration in durations}

# # Calcular las precipitaciones máximas para cada duración
# for duration in durations:
#     max_precipitations[duration] = df.iloc[:, 1:].max(axis=1).rolling(window=duration).max().dropna()

# # Ajustar una distribución de Gumbel a los datos
# gumbel_params = {duration: gumbel_r.fit(max_precipitations[duration]) for duration in durations}

# # Definir períodos de retorno (en años)
# return_periods = [2, 5, 10, 25, 50, 100]

# # Calcular las precipitaciones para cada duración y período de retorno
# ddf_values = {duration: {} for duration in durations}
# for duration in durations:
#     for rp in return_periods:
#         exceedance_probability = 1 / rp
#         ddf_values[duration][rp] = gumbel_r.ppf(1 - exceedance_probability, *gumbel_params[duration])

# # Graficar las curvas DDF
# plt.figure(figsize=(12, 8))
# for rp in return_periods:
#     plt.plot(durations, [ddf_values[duration][rp] for duration in durations], label=f'{rp}-Year Return Period')

# plt.xlabel('Duration (minutes)')
# plt.ylabel('Precipitation Depth (mm)')
# plt.title('Depth-Duration-Frequency Curves')
# plt.xscale('log')
# plt.yscale('log')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', linewidth=0.5)
# plt.show()
