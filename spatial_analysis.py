#%%
# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import os
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import cKDTree
from scipy.interpolate import Rbf

#%%

# Open the.shp
rainfall_2002 = r"C:\Users\Aldis\Documents\Master Data Science\GitHub\Urban-Hydrology-Rainfall-Models-and-Simulations\Data\Spatial_Analysis\hourly_rainfall_shape_2002.shp"
# Read the shapefile
rainfall_2002 = gpd.read_file(rainfall_2002)
# Plot
rainfall_2002.plot()
# Print Plot
plt.title("Map of Rainfall_2002")
plt.show()
#%%

# Loading the Arno River Basin Shapefile
basin_bounds = r"C:\Users\Aldis\Documents\Master Data Science\GitHub\Urban-Hydrology-Rainfall-Models-and-Simulations\Data\Spatial_Analysis\limite_adb_arno.shp"
bb = gpd.read_file(basin_bounds)
bb.plot()
plt.title("Arno River Bounds")
plt.show()
#%%

# Select a trial day (25 de diciembre de 2002)
selected_day = "2002-12-25"  
# Filter the trial day
rainfall_2002_selected = rainfall_2002[rainfall_2002['trial_day'] == selected_day]
# Plot the trial day data in the map bounder 
fig, ax = plt.subplots(figsize=(10, 8))
bb.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.5) 
rainfall_2002.plot(ax=ax, column='trial_day', cmap='RdBu', markersize=5, legend=True)  
rainfall_2002_selected.plot(ax=ax, color='red', markersize=50, label=f'Selected Day ({selected_day})')  

plt.title(f'Sampled precipitation (in mm) - Selected Day: {selected_day}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(['Basin Bounds', 'Trial Day', f'Selected Day ({selected_day})'], loc='upper right')
plt.tight_layout()
plt.show()
# %%

# Construct Thiessen polygons and set the projection system to the same as the rainfall dataset
points = np.array([[point.x, point.y] for point in rainfall_2002.geometry])
# Calculate points
vor = Voronoi(points)
# Plot points
fig, ax = plt.subplots(figsize=(10, 8))
# Plot bounds
bb.plot(ax=ax, color='lightblue', edgecolor='black', alpha=0.5)
# Plot polygons
voronoi_plot_2d(vor, ax=ax, show_vertices=False)
# Title
plt.title('Thiessen Polygons for Interpolation')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(['Basin Bounds', 'Thiessen Polygons'], loc='upper right')
plt.tight_layout()
plt.show()
# %%

# Interpolation Data
# Path to the shapefile containing rainfall data
# rainfall_2002 = r"C:\Users\Aldis\Documents\Master Data Science\GitHub\Urban-Hydrology-Rainfall-Models-and-Simulations\Data\Spatial_Analysis\hourly_rainfall_shape_2002.shp"
# # Read the shapefile
# rainfall_2002 = gpd.read_file(rainfall_2002)
# Create a regular grid (100x100)for IDW interpolation
# Adjust the number of points as needed
n_points = 100
x_min, x_max, y_min, y_max = rainfall_2002_selected.total_bounds
x = np.linspace(x_min, x_max, n_points)
y = np.linspace(y_min, y_max, n_points)
xx, yy = np.meshgrid(x, y)
grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
# Convert to GeoDataFrame
grid_gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy(grid_points[:, 0], grid_points[:, 1]), crs=rainfall_2002.crs)

# Perform IDW interpolation
# Use cKDTree to find nearest neighbors and calculate weights
tree = cKDTree(rainfall_2002_selected.geometry.apply(lambda geom: (geom.x, geom.y)).tolist())
distances, indices = tree.query(grid_points, k=10)  # k = number of neighbors to consider
weights = 1.0 / distances**2
weights /= np.sum(weights, axis=1)[:, np.newaxis]
interpolated_values = np.sum(weights * rainfall_2002_selected['trial_day'].values[indices], axis=1)
# Add interpolated values to the GeoDataFrame grid
grid_gdf['interpolated_values'] = interpolated_values

# Plot the IDW interpolation results
fig, ax = plt.subplots(figsize=(10, 8))
# Plot original data points
rainfall_2002.plot(ax=ax, column='trial_day', cmap='RdBu', markersize=5, legend=True)
# Plot grid with interpolated values
grid_gdf.plot(ax=ax, column='interpolated_values', cmap='RdBu', markersize=1, legend=True)
plt.title('IDW Interpolation of Rainfall Data')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()
# %%

