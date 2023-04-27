import numpy as np

# for exp in [2]:
#     shap = [np.load(f'imdb_genre_{exp}_shap_values_{i}.npy') for i in range(0,200,10)]
#     shap = np.concatenate(shap, axis=1)
#     np.save(f'imdb_genre_{exp}_shap_values.npy', shap)
    
for exp in [0]:
    shap = [np.load(f'imdb_genre_{exp}_val_shap_values_{i}.npy') for i in range(0,120,10)]
    shap = np.concatenate(shap, axis=1)
    np.save(f'imdb_genre_{exp}_val_shap_values.npy', shap)