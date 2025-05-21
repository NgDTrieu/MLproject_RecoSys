import os
import sys
import numpy as np
import pandas as pd
from surprise import Reader, Dataset, CoClustering, accuracy
from surprise.model_selection import cross_validate
import itertools

# Load ratings DF
ratings = pd.read_pickle('/kaggle/input/mlproject/books2rec/.tmp/ratings_pickle')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userID', 'itemID', 'rating']], reader)

# Parameter grid for CoClustering
param_grid = {
    'n_cltr_u': [2, 3, 4],
    'n_cltr_i': [2, 3, 4],
    'n_epochs': [10, 20, 30]
}
keys, values = zip(*param_grid.items())
params = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_RMSE = float('inf')
best_param = None
results_data = []

for param in params:
    print("Running CoClustering with params: %s" % param)
    algo = CoClustering(n_cltr_u=param['n_cltr_u'], n_cltr_i=param['n_cltr_i'], 
                        n_epochs=param['n_epochs'])
    
    # Run 5-fold cross-validation
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, n_jobs=-1, verbose=True)
    
    avg_rmse = np.mean(results['test_rmse'])
    results_data.append({
        'params': param,
        'rmse': avg_rmse,
        'n_cltr_u': param['n_cltr_u'],
        'n_cltr_i': param['n_cltr_i'],
        'n_epochs': param['n_epochs']
    })
    
    if avg_rmse < best_RMSE:
        best_RMSE = avg_rmse
        best_param = param

# Print results
print("\nBest RMSE: %s" % best_RMSE)
print("Best params: %s" % best_param)

# Convert results to DataFrame and display for Kaggle
results_df = pd.DataFrame(results_data)
print("\nGrid Search Results:")
print(results_df[['n_cltr_u', 'n_cltr_i', 'n_epochs', 'rmse']])

# Save results to a CSV file for visualization in Kaggle
results_df.to_csv('coclustering_grid_search_results.csv', index=False)
print("\nResults saved to 'coclustering_grid_search_results.csv' for visualization.")