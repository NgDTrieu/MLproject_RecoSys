import os
import sys
import numpy as np
import pandas as pd
from surprise import Reader, Dataset, BaselineOnly, accuracy
from surprise.model_selection import cross_validate
import itertools

# Load ratings DF
ratings = pd.read_pickle('/kaggle/input/mlproject/books2rec/.tmp/ratings_pickle')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userID', 'itemID', 'rating']], reader)

# Parameter grid for BaselineOnly
param_grid = {
    'bsl_options': [
        {'method': 'als', 'n_epochs': 5, 'reg_u': 10, 'reg_i': 10},
        {'method': 'als', 'n_epochs': 10, 'reg_u': 10, 'reg_i': 10},
        {'method': 'als', 'n_epochs': 20, 'reg_u': 10, 'reg_i': 10},
        {'method': 'als', 'n_epochs': 10, 'reg_u': 5, 'reg_i': 10},
        {'method': 'als', 'n_epochs': 10, 'reg_u': 15, 'reg_i': 10},
        {'method': 'sgd', 'n_epochs': 5, 'learning_rate': 0.005},
        {'method': 'sgd', 'n_epochs': 10, 'learning_rate': 0.005},
        {'method': 'sgd', 'n_epochs': 20, 'learning_rate': 0.005},
        {'method': 'sgd', 'n_epochs': 10, 'learning_rate': 0.001},
        {'method': 'sgd', 'n_epochs': 10, 'learning_rate': 0.01}
    ]
}
keys, values = zip(*param_grid.items())
params = [dict(zip(keys, v)) for v in itertools.product(*values)]

best_RMSE = float('inf')
best_param = None
results_data = []

for param in params:
    print("Running BaselineOnly with params: %s" % param)
    algo = BaselineOnly(bsl_options=param['bsl_options'])
    
    # Run 5-fold cross-validation
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, n_jobs=-1, verbose=True)
    
    avg_rmse = np.mean(results['test_rmse'])
    results_data.append({
        'params': param['bsl_options'],
        'rmse': avg_rmse,
        'method': param['bsl_options']['method'],
        'n_epochs': param['bsl_options']['n_epochs']
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
print(results_df[['method', 'n_epochs', 'rmse']])

# Save results to a CSV file for visualization in Kaggle
results_df.to_csv('baseline_grid_search_results.csv', index=False)
print("\nResults saved to 'baseline_grid_search_results.csv' for visualization.")