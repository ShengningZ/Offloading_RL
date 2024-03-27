import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('training_results_20240327_213132.csv')

# Calculate the total latency and energy for each configuration
data['total_latency'] = data.groupby('config')['total_latency'].transform('sum')
data['total_energy'] = data.groupby('config')['total_energy'].transform('sum')

# Get the unique configurations
configs = data['config'].unique()

# Create a new DataFrame to store the best configuration for each iteration
best_configs = pd.DataFrame(columns=['iteration', 'best_config', 'total_latency', 'total_energy'])

# Find the best configuration for each iteration
for iteration in data['iteration'].unique():
    iteration_data = data[data['iteration'] == iteration]
    best_config = iteration_data.loc[iteration_data['reward'].idxmax()]['config']
    total_latency = iteration_data.loc[iteration_data['config'] == best_config, 'total_latency'].values[0]
    total_energy = iteration_data.loc[iteration_data['config'] == best_config, 'total_energy'].values[0]
    new_row = pd.DataFrame({'iteration': [iteration], 'best_config': [best_config],
                            'total_latency': [total_latency], 'total_energy': [total_energy]})
    best_configs = pd.concat([best_configs, new_row], ignore_index=True)

# Plot the total latency and energy for each configuration
plt.figure(figsize=(12, 6))
for config in configs:
    config_data = data[data['config'] == config]
    plt.scatter(config_data['total_latency'], config_data['total_energy'], label=config, alpha=0.7)
plt.xlabel('Total Latency (s)')
plt.ylabel('Total Energy (J)')
plt.title('Total Latency and Energy for Each Configuration')
plt.legend()
plt.tight_layout()
plt.show()

# Plot the best configuration for each iteration
plt.figure(figsize=(12, 6))
for config in configs:
    config_data = best_configs[best_configs['best_config'] == config]
    plt.scatter(config_data['iteration'], config_data['total_latency'], label=config, alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Total Latency (s)')
plt.title('Best Configuration - Total Latency')
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
for config in configs:
    config_data = best_configs[best_configs['best_config'] == config]
    plt.scatter(config_data['iteration'], config_data['total_energy'], label=config, alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Total Energy (J)')
plt.title('Best Configuration - Total Energy')
plt.legend()
plt.tight_layout()
plt.show()