import surprise
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD, dump, accuracy
from surprise.model_selection import GridSearchCV, cross_validate
from collections import defaultdict
import itertools

ratings = pd.read_pickle('ratings_pickle')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['userID', 'itemID', 'rating']], reader)


# In[ ]:


param_grid = {'n_epochs': [20, 50, 100], 
              'n_factors': [10, 20, 50],
              'lr_all': [0.005],
              'reg_all': [0.02]}
keys, values = zip(*param_grid.items())
params = [dict(zip(keys, v)) for v in itertools.product(*values)]
print(params)

# In[10]:


#param_grid = {'n_epochs': [1], 
#              'n_factors': [3],
#              'lr_all': [0.5],
#              'reg_all': [0.52]}
#keys, values = zip(*param_grid.items())
#params = [dict(zip(keys, v)) for v in itertools.product(*values)]


# In[12]:


best_RMSE = float('inf')
best_param = None
for param in params:
    print("Running SVD with params: %s" % (param))
    algo = SVD(lr_all=param['lr_all'],reg_all=param['reg_all'], n_epochs=param['n_epochs'],n_factors=param['n_factors'])
    
    # Run 5-fold cross-validation and print results
    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, n_jobs=-1, verbose=True)
    
    avg_rmse = 0
    rmse = results['test_rmse']
    for r in rmse:
        avg_rmse += r
    avg_rmse /= len(rmse)
    
    if avg_rmse < best_RMSE:
        best_RMSE = avg_rmse
        best_param = param


# In[ ]:


print("Best RMSE: %s" % best_RMSE)
print("Best params: %s" % best_param)


# In[16]:


with open('../.tmp/grid_search_best_params.txt', 'w') as the_file:
    the_file.write("Best RMSE: %s\n" % best_RMSE)
    the_file.write("Best params: %s" % best_param)
