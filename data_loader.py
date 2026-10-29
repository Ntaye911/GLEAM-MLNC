import os
import numpy as np
import pandas as pd
from scipy.io import arff

def load_dataset(dataset_name, data_dir="datasets"):
    """
    Loads a multi-label dataset from the specified directory.
    
    folder_path = os.path.join(data_dir, dataset_name)
    
    
    files = os.listdir(folder_path)
    mat_files = [f for f in files if f.endswith('.mat')]
    mat_files = [f for f in files if f.endswith('.mat')]
    
    if mat_files:
        # Assume the first .arff file contains both features and labels
        # MULAN format: features first, then labels. 
        # You may need to adjust `num_labels` based on the specific dataset.
        file_path = os.path.join(folder_path, arff_files[0])
        data, meta = arff.loadarff(file_path)
        df = pd.DataFrame(data)
        
        # Convert categorical/bytes to numeric
        df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
        
       # Adjust this slice based on your dataset's specific structure.
       
        num_labels = 14 # <-- UPDATE THIS PER DATASET OR PARSE FROM META
        X = df.iloc[:, :-num_labels].values.astype(float)
        Y = df.iloc[:, -num_labels:].values.astype(float)
        
    elif csv_files:
        file_path = os.path.join(folder_path, csv_files[0])
        df = pd.read_csv(file_path)
        # Assume last 'c' columns are labels. Update 'num_labels' accordingly.
        num_labels = 14 # <-- UPDATE THIS PER DATASET
        X = df.iloc[:, :-num_labels].values.astype(float)
        Y = df.iloc[:, -num_labels:].values.astype(float)
    else:
        raise FileNotFoundError(f"No .arff or .csv files found in {folder_path}")
    
    # Normalize features to unit L2 norm as per paper
    from sklearn.preprocessing import normalize
    X = normalize(X, norm='l2', axis=1)
    
    return X, Y

if __name__ == "__main__":
    # Test loader
    X, Y = load_dataset("yeast")
    print(f"Loaded X shape: {X.shape}, Y shape: {Y.shape}")