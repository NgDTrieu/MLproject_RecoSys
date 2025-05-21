import os
import sys
import numpy as np
import pandas as pd
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import cross_validate
import itertools

# Load ratings DF
ratings = pd.read_pickle('/kaggle/input/mlproject/books2rec/.tmp/ratings_pickle')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userID', 'itemID', 'rating']], reader)

# Parameter grid for SVD
param_grid = {
    'n_epochs': [20, 50, 100],
    'n_factors': [10, 300, 1000]
}
keys, values = zip(*param_grid.items())
params = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_RMSE = float('inf')
best_param = None
results_data = []

for param in params:
    print("Running SVD with params: %s" % param)
    algo = SVD(n_epochs=param['n_epochs'], n_factors=param['n_factors'], lr_all=0.005, reg_all=0.02)
    
    # Run 5-fold cross-validation
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, n_jobs=-1, verbose=True)
    
    avg_rmse = np.mean(results['test_rmse'])
    results_data.append({
        'params': param,
        'rmse': avg_rmse,
        'n_epochs': param['n_epochs'],
        'n_factors': param['n_factors']
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
print(results_df[['n_epochs', 'n_factors', 'rmse']])

# Save results to a CSV file for visualization in Kaggle
results_df.to_csv('svd_grid_search_results.csv', index=False)
print("\nResults saved to 'svd_grid_search_results.csv' for visualization.")