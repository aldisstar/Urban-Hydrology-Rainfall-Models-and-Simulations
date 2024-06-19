#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gamma, expon, lognorm, kstest
from scipy.stats import ks_2samp
from scipy.stats import genextreme
import statsmodels.api as sm
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.dates as mdates

print("All packages imported successfully!")
#%%

# Import data
daily_data = pd.read_csv("C:/Users/Aldis/Documents/Master Data Science/NYU/Rainfall_Daily_Data.csv")
#%%

# Select the first 2 columns
Daily_Ts = daily_data.iloc[:, [0, 11]]
Daily_Ts.columns = ['Time', 'Pluv_10']

# Convert character to a "Date" object
Daily_Ts['Time'] = pd.to_datetime(Daily_Ts['Time'])
print(Daily_Ts)
#%%

# Plot time series
plt.figure(figsize=(10, 6))
plt.plot(Daily_Ts['Time'], Daily_Ts['Pluv_10'], label='Observed Daily Ts')
plt.title('Observed Daily Ts')
plt.xlabel('Time')
plt.ylabel('mm')
plt.grid(True)
plt.show()
#%%

# Qualitative analysis
rainfall_df = pd.DataFrame({'rainfall': Daily_Ts['Pluv_10']})

# Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x=rainfall_df['rainfall'], color='steelblue')
plt.title('Boxplot of Rainfall Data')
plt.xlabel('Value (mm)')
plt.show()
#%%

# Histogram
plt.figure(figsize=(10, 6))
sns.histplot(rainfall_df['rainfall'], bins=int(10), color='steelblue')
plt.title('Frequency Histogram')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()

# Quantitative analysis
# Approch 1 Summary statistics
print(rainfall_df['rainfall'].describe())
#%%

# Approach 2 Detailed statistics
# Define the describe function with additional percentiles
def describe(data):
    desc = data.describe()
    percentiles = data.quantile([0.95, 0.975, 0.99, 0.999])
    percentiles.index = ['95%', '97.5%', '99%', '99.9%']
    percentiles = percentiles.T
    desc = pd.concat([desc, percentiles])
    return desc

# Use the describe function on the 'Pluv_10' column
ss = describe(Daily_Ts['Pluv_10'])

# Select specific columns to display (note: converting to dataframe to select columns as in R)
ss_selected = ss[['count', 'mean', 'std', '50%', 'min', '95%', '97.5%', '99%', '99.9%', 'max']]
print(ss_selected)
#%%

# ECDF plot
rainfall_data = rainfall_df['rainfall'].dropna()
ecdf = sm.distributions.ECDF(rainfall_data)
x = np.linspace(min(rainfall_data), max(rainfall_data), num=100)
y = ecdf(x)

plt.figure(figsize=(10, 6))
plt.step(x, y, where='post')
plt.title('ECDF Plot')
plt.xlabel('mm')
plt.ylabel('ECDF')
plt.grid(True)
plt.show()
#%%

# ACF plot
fig, ax = plt.subplots(figsize=(10, 6))
sm.graphics.tsa.plot_acf(rainfall_data, lags=5, ax=ax)
plt.show()
#%%

# Return Period Estimation - Empirical approach
# Function to extract annual maxima
def annual_maxima(df):
    df = df.copy()  # Avoid SettingWithCopyWarning
    df['Year'] = df['Time'].dt.year
    ann_max = df.groupby('Year')['Pluv_10'].max().reset_index()
    return ann_max

# Apply the function to extract annual maxima
ann_max = annual_maxima(Daily_Ts)

# Calculate ECDF for annual maxima
ecdf = ECDF(ann_max['Pluv_10'])

# Calculate RP for a given peak value using ECDF
peak_value = 78
RP_ecdf = 1 / (1 - ecdf(peak_value))
print(f'The RP calculated with ECDF is: {RP_ecdf} years')

# Calculate RP using GEV distribution
# Fit GEV distribution
params = genextreme.fit(ann_max['Pluv_10'])
shape, loc, scale = params

# Calculate RP using GEV (Generalize Extream Values) distribution
RP_gev = 1 / (1 - genextreme.cdf(peak_value, shape, loc=loc, scale=scale))
print(f'The RP calculated with the GEV distribution is: {RP_gev} years')
#%%


# Fitting multiple distributions on the data
# Fit gamma, exponential, and log-normal distributions
params_gamma = gamma.fit(rainfall_df['rainfall'])
params_exp = expon.fit(rainfall_df['rainfall'])
params_lognorm = lognorm.fit(rainfall_df['rainfall'][rainfall_df['rainfall'] > 0], floc=0)

# Create a sequence of values for the x-axis
x = np.linspace(min(rainfall_df['rainfall']), max(rainfall_df['rainfall']), 100)

# Calculate PDFs
pdf_gamma = gamma.pdf(x, *params_gamma)
pdf_exp = expon.pdf(x, *params_exp)
pdf_lognorm = lognorm.pdf(x, *params_lognorm)

# Plot the histogram and the fitted PDFs for each distribution
plt.figure(figsize=(18, 12))

# Plot for Gamma distribution
plt.subplot(3, 1, 1)
sns.histplot(rainfall_df['rainfall'], bins=25, kde=False, stat='density', color='steelblue', alpha=0.6)
plt.plot(x, pdf_gamma, label='Gamma', color='red')
plt.title('Fitting Gamma Distribution to Rainfall Data')
plt.xlabel('Rainfall Amount (mm)')
plt.ylabel('Density')
plt.legend()

# Plot for Exponential distribution
plt.subplot(3, 1, 2)
sns.histplot(rainfall_df['rainfall'], bins=25, kde=False, stat='density', color='steelblue', alpha=0.6)
plt.plot(x, pdf_exp, label='Exponential', color='blue')
plt.title('Fitting Exponential Distribution to Rainfall Data')
plt.xlabel('Rainfall Amount (mm)')
plt.ylabel('Density')
plt.legend()

# Plot for Log-Normal distribution
plt.subplot(3, 1, 3)
sns.histplot(rainfall_df['rainfall'], bins=25, kde=False, stat='density', color='steelblue', alpha=0.6)
plt.plot(x, pdf_lognorm, label='Log-Normal', color='orange')
plt.title('Fitting Log-Normal Distribution to Rainfall Data')
plt.xlabel('Rainfall Amount (mm)')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()
#%%


# Goodness of fit tests: Quantitative analysis
# Remove zero or negative values for gamma and log-normal fitting
positive_rainfall = rainfall_df['rainfall'][rainfall_df['rainfall'] > 0]

# Fitting multiple distributions on the data
# Fit gamma, exponential, and log-normal distributions
params_exp_mle = expon.fit(rainfall_df['rainfall'])
params_gamma_mle = gamma.fit(positive_rainfall)
params_lognorm_mle = lognorm.fit(positive_rainfall, floc=0)

# Method of Moments (MM)
# Exponential MM: mean
mean_exp = np.mean(rainfall_df['rainfall'])
params_exp_mm = (0, mean_exp)

# Gamma MM: mean and variance
mean_gamma = np.mean(positive_rainfall)
var_gamma = np.var(positive_rainfall)
shape_gamma_mm = mean_gamma**2 / var_gamma
scale_gamma_mm = var_gamma / mean_gamma
params_gamma_mm = (shape_gamma_mm, 0, scale_gamma_mm)

# Log-normal MM: mean and variance of log-transformed data
log_data = np.log(positive_rainfall)
mean_lognorm = np.mean(log_data)
std_lognorm = np.std(log_data)
params_lognorm_mm = (std_lognorm, 0, np.exp(mean_lognorm))

# Goodness of fit tests
ks_exp_mle = kstest(rainfall_df['rainfall'], 'expon', args=params_exp_mle)
ks_exp_mm = kstest(rainfall_df['rainfall'], 'expon', args=params_exp_mm)

ks_gamma_mle = kstest(positive_rainfall, 'gamma', args=params_gamma_mle)
ks_gamma_mm = kstest(positive_rainfall, 'gamma', args=params_gamma_mm)

ks_lognorm_mle = kstest(positive_rainfall, 'lognorm', args=params_lognorm_mle)
ks_lognorm_mm = kstest(positive_rainfall, 'lognorm', args=params_lognorm_mm)

print(f'KS test for Exponential (MLE): {ks_exp_mle}')
print(f'KS test for Exponential (MM): {ks_exp_mm}')

print(f'KS test for Gamma (MLE): {ks_gamma_mle}')
print(f'KS test for Gamma (MM): {ks_gamma_mm}')

print(f'KS test for Log-Normal (MLE): {ks_lognorm_mle}')
print(f'KS test for Log-Normal (MM): {ks_lognorm_mm}')

# Plotting PDFs for comparison
x = np.linspace(min(positive_rainfall), max(positive_rainfall), 100)

pdf_exp_mle = expon.pdf(x, *params_exp_mle)
pdf_exp_mm = expon.pdf(x, *params_exp_mm)

pdf_gamma_mle = gamma.pdf(x, *params_gamma_mle)
pdf_gamma_mm = gamma.pdf(x, *params_gamma_mm)

pdf_lognorm_mle = lognorm.pdf(x, *params_lognorm_mle)
pdf_lognorm_mm = lognorm.pdf(x, *params_lognorm_mm)

plt.figure(figsize=(18, 12))

# Plot for Exponential distribution
plt.subplot(3, 1, 1)
sns.histplot(rainfall_df['rainfall'], bins=25, kde=False, stat='density', color='steelblue', alpha=0.6)
plt.plot(x, pdf_exp_mle, label='Exponential MLE', color='red')
plt.plot(x, pdf_exp_mm, label='Exponential MM', color='blue')
plt.title('Fitting Exponential Distribution to Rainfall Data')
plt.xlabel('Rainfall Amount (mm)')
plt.ylabel('Density')
plt.legend()

# Plot for Gamma distribution
plt.subplot(3, 1, 2)
sns.histplot(positive_rainfall, bins=25, kde=False, stat='density', color='steelblue', alpha=0.6)
plt.plot(x, pdf_gamma_mle, label='Gamma MLE', color='red')
plt.plot(x, pdf_gamma_mm, label='Gamma MM', color='blue')
plt.title('Fitting Gamma Distribution to Rainfall Data')
plt.xlabel('Rainfall Amount (mm)')
plt.ylabel('Density')
plt.legend()

# Plot for Log-Normal distribution
plt.subplot(3, 1, 3)
sns.histplot(positive_rainfall, bins=25, kde=False, stat='density', color='steelblue', alpha=0.6)
plt.plot(x, pdf_lognorm_mle, label='Log-Normal MLE', color='red')
plt.plot(x, pdf_lognorm_mm, label='Log-Normal MM', color='blue')
plt.title('Fitting Log-Normal Distribution to Rainfall Data')
plt.xlabel('Rainfall Amount (mm)')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()
#%%
