import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

df = pd.read_excel('info.xlsx')


# Set font for plots (English default)
plt.rcParams['axes.unicode_minus'] = False

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))

# Plot scatter with jitter (to show overlapping points)
x_jitter = df['True Values'] + np.random.normal(0, 0.08, size=len(df))
y_jitter = df['Predicted Values'] + np.random.normal(0, 0.08, size=len(df))

ax.scatter(x_jitter, y_jitter, 
           alpha=0.5, s=40, label='Measured vs Predicted')

# Linear regression
x = df['True Values']
y = df['Predicted Values']
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

# Calculate metrics
r2 = r_value**2
mae = mean_absolute_error(x, y)
rmse = np.sqrt(mean_squared_error(x, y))

# Plot fitting line
x_range = np.linspace(x.min(), x.max(), 100)
y_range = slope * x_range + intercept
ax.plot(x_range, y_range, 'r-', linewidth=2, label=f'Fitting line: y = {slope:.2f}x + {intercept:.2f}')

# Calculate 95% Confidence Interval for the regression line
n = len(x)
dof = n - 2
t_val = stats.t.ppf(0.975, dof)

# Standard error of the estimate (residual standard error)
residual = y - (slope * x + intercept)
s_err = np.sqrt(np.sum(residual**2) / dof)

# CI calculation
x_mean = np.mean(x)
ssx = np.sum((x - x_mean)**2)
ci = t_val * s_err * np.sqrt(1/n + (x_range - x_mean)**2 / ssx)

ax.fill_between(x_range, y_range - ci, y_range + ci, color='r', alpha=0.15, label='95% Confidence Interval')

# 1:1 line
ax.plot([x.min(), x.max()], [x.min(), x.max()], 'k--', alpha=0.5, label='1:1 Line')

# Annotation
stats_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nn = {len(x)}'
ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
       fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')
ax.set_title('True Values vs Predicted Values')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()