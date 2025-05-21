import pandas as pd
import matplotlib.pyplot as plt

# Load results from CSV
results_df = pd.read_csv('svd_grid_search_results.csv')

# Plot RMSE vs n_epochs for each n_factors
plt.figure(figsize=(10, 6))
for n_factors in results_df['n_factors'].unique():
    data = results_df[results_df['n_factors'] == n_factors]
    plt.plot(data['n_epochs'], data['rmse'], marker='o', label=f'Số chiều ẩn = {n_factors}')

plt.xlabel('Số Epoch')
plt.ylabel('RMSE')
plt.title('So sánh RMSE theo Số Epoch cho SVD với các Số Chiều Ẩn')
plt.grid(True)
plt.legend()
plt.savefig('svd_rmse_comparison.png')
plt.show()