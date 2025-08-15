import os
import pandas as pd
import numpy as np

def num_files(data_path):
    '''
    Gets number of files in a data path for each class.
    
    args:
        data_path (str): path to data
    returns:
        df (pd.DataFrame): dataframe with number of files for each class
    '''
    
    df = []
    
    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path, class_name)
        if os.path.isdir(class_path):
            num_files = len(os.listdir(class_path))
            df.append((class_name, num_files))

    return pd.DataFrame(df, columns=['Class', 'Num Files'])