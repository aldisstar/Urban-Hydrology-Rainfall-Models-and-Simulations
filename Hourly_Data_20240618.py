#%%
#Librerias
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress

# Importar datos
subhourly_data = pd.read_csv("C:/Users/Aldis/Documents/Master Data Science/NYU/Rainfall_Sub_hourly_Data.csv")

# Seleccionar las columnas necesarias
SubH_Ts = subhourly_data.iloc[:, [0, 11]]

# Renombrar las columnas
SubH_Ts.columns = ['Time', 'Precip']

# Convertir la columna 'Time' a un objeto datetime
SubH_Ts['Time'] = pd.to_datetime(SubH_Ts['Time'], format="%Y-%m-%d %H:%M:%S")

# Definir la función idf_ddf_intermittency_n
def idf_ddf_intermittency_n(Ts, n_steps):
    # Rename columns
    Ts.columns = ['Time', 'Precip']
    
    # Compute the number of years
    n_years = len(Ts['Time'].dt.year.unique())
    
    ## IDF Curves
    # Create temporary DataFrame for IDF curves
    Ts_temp = Ts.copy()
    Ts_temp['A_Ts_1h'] = Ts_temp['Precip'].rolling(window=1*n_steps, min_periods=1).sum() / (1*n_steps)
    Ts_temp['B_Ts_3h'] = Ts_temp['Precip'].rolling(window=3*n_steps, min_periods=1).sum() / (3*n_steps)
    Ts_temp['C_Ts_6h'] = Ts_temp['Precip'].rolling(window=6*n_steps, min_periods=1).sum() / (6*n_steps)
    Ts_temp['D_Ts_12h'] = Ts_temp['Precip'].rolling(window=12*n_steps, min_periods=1).sum() / (12*n_steps)
    Ts_temp['E_Ts_24h'] = Ts_temp['Precip'].rolling(window=24*n_steps, min_periods=1).sum() / (24*n_steps)
    
    # Add 'Year' column and set as factor
    Ts_temp['Year'] = Ts_temp['Time'].dt.year.astype('category')
    
    # Calculate annual maxima
    ann_max_long_df = pd.melt(Ts_temp, id_vars=['Year'], value_vars=['A_Ts_1h', 'B_Ts_3h', 'C_Ts_6h', 'D_Ts_12h', 'E_Ts_24h'],
                              var_name='Duration', value_name='Max').groupby(['Duration', 'Year']).max().reset_index()
    
    # Convert from long to wide format
    ann_max_wide_df = ann_max_long_df.pivot(index='Year', columns='Duration', values='Max')
    
    # Sort IDF values in descending order
    idf = ann_max_wide_df.apply(lambda x: np.sort(x)[::-1], axis=0)
    
    # Plot IDF curves
    idf_plot = plt.figure(figsize=(10, 6))
    for col in idf.columns:
        plt.plot(idf.index, idf[col], marker='o', label=col)
    plt.xlabel('Time [hours]')
    plt.ylabel('mm')
    plt.title('Intensity-Duration-Frequency curves')
    plt.legend(title='Duration')
    plt.grid(True)
    plt.show()
    
    ## DDF Curves
    # Create temporary DataFrame for DDF curves
    Ts_temp2 = Ts.copy()
    Ts_temp2['A_Ts_1h'] = Ts_temp2['Precip'].rolling(window=1*n_steps, min_periods=1).sum()
    Ts_temp2['B_Ts_3h'] = Ts_temp2['Precip'].rolling(window=3*n_steps, min_periods=1).sum()
    Ts_temp2['C_Ts_6h'] = Ts_temp2['Precip'].rolling(window=6*n_steps, min_periods=1).sum()
    Ts_temp2['D_Ts_12h'] = Ts_temp2['Precip'].rolling(window=12*n_steps, min_periods=1).sum()
    Ts_temp2['E_Ts_24h'] = Ts_temp2['Precip'].rolling(window=24*n_steps, min_periods=1).sum()
    
    # Add 'Year' column and set as factor
    Ts_temp2['Year'] = Ts_temp2['Time'].dt.year.astype('category')
    
    # Calculate annual maxima for DDF
    ann_max_long_df2 = pd.melt(Ts_temp2, id_vars=['Year'], value_vars=['A_Ts_1h', 'B_Ts_3h', 'C_Ts_6h', 'D_Ts_12h', 'E_Ts_24h'],
                                var_name='Duration', value_name='Max').groupby(['Duration', 'Year']).max().reset_index()
    
    # Convert from long to wide format for DDF
    ann_max_wide_df2 = ann_max_long_df2.pivot(index='Year', columns='Duration', values='Max')
    
    # Sort DDF values in descending order
    ddf = ann_max_wide_df2.apply(lambda x: np.sort(x)[::-1], axis=0)
    
    # Plot DDF curves
    ddf_plot = plt.figure(figsize=(10, 6))
    for col in ddf.columns:
        plt.plot(ddf.index, ddf[col], marker='o', label=col)
    plt.xlabel('Time [hours]')
    plt.ylabel('mm')
    plt.title('Depth-Duration-Frequency curves')
    plt.legend(title='Duration')
    plt.grid(True)
    plt.show()
    
    # Estimate the slope parameter n
    n_par_df = pd.DataFrame(np.zeros((n_years, 1)), columns=['n_par'])
    durations = [1, 3, 6, 12, 24]
    log_durations = np.log(durations)
    
    for i, year in enumerate(ddf.index):
        log_ddf = np.log(ddf.iloc[i, :])
        slope, intercept, r_value, p_value, std_err = linregress(log_durations, log_ddf)
        n_par_df.iloc[i, 0] = slope
    
    n_par = n_par_df['n_par'].mean()
    
    # Intermittency
    dry_freq = len(Ts[Ts['Precip'] == 0]) / len(Ts)
    
    return idf_plot, ddf_plot, dry_freq, n_par

# Ejecutar la función con los datos SubH_Ts y n_steps = 4
idf_plot, ddf_plot, dry_freq, n_par = idf_ddf_intermittency_n(SubH_Ts, n_steps=4)

# Mostrar los resultados
print("Dry Frequency:", dry_freq)
print("n Parameter:", n_par)
