# -*- coding: utf-8 -*-
"""
Created on Wed May 21 12:32:05 2025

@author: Ahmed H. Hanfy
"""
import os
import sys
import json
import numpy as np
import LDA_statistics_v2 as LDA

def loading_data(data_file):
    with open(data_file, 'r') as f:
        loaded_data = json.load(f)
        
    print(f"Loaded structure type: {type(loaded_data)}")
    All_arrayes=list(loaded_data.keys())
    print(f"Available keys (array names): {All_arrayes}")
    print("-" * 30)
    
    if 'suffixes_set' in All_arrayes:
        suffixes_set = loaded_data['suffixes_set']
        print("Array suffixes_set is loaded")
    else: 
        print("Error: suffixes_set is not exist!")
        sys.exit()
        
    num_comps = loaded_data['num_comps'] if 'num_comps' in All_arrayes else 1   
    vx_filter_range = loaded_data['vx_filter_range'] if 'vx_filter_range' in All_arrayes else None
    vy_filter_range = loaded_data['vy_filter_range'] if 'vy_filter_range' in All_arrayes else None
    wy = loaded_data['wy'] if 'wy' in All_arrayes else None
    wx = loaded_data['wx'] if 'wx' in All_arrayes else None
    
    
    return suffixes_set, num_comps, vx_filter_range, vy_filter_range, wy, wx, 

if __name__ == '__main__':
    # Define your inputs
    directory = r'folderdirectory'
    base_file = 'FileBaseName.0000'
    
    try:
        data_file= fr'{directory}/info.json'
        if os.path.exists(data_file):
            print(f"The file '{data_file}' exists.")
            load_data = input("Would you like to load it? Type Y if yes: ").strip()
            if load_data[0].upper() == 'Y': 
                suffixes_set, num_comps, vx_filter_range, vy_filter_range, wy, wx = loading_data(data_file)
    except IOError as e:
        print(f"Error writing to file {data_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")  
    
    Data_set = []
    output_filename_np = 'Processed data.csv'
      
    
    for i, suffixes in enumerate(suffixes_set):    
        # Call the function
        n_comps=num_comps[i] if hasattr(num_comps, "__len__") else num_comps
        vx_range=vx_filter_range[i] if hasattr(vx_filter_range, "__len__") else vx_filter_range
        GMMx_weights = wx[i] if hasattr(wx, "__len__") else wx
        statistics_output = LDA.analyze_lda_data(
                                                 directory_path=directory,
                                                 main_file_name=base_file,
                                                 file_suffixes=suffixes,
                                                 num_components=n_comps,
                                                 vx_range=vx_range,
                                                 vy_range=vy_filter_range,
                                                 GMMx_weights = GMMx_weights,
                                                 GMMy_weights = wy
                                                )
        Data_set.append(statistics_output)
        
    numpy_array = np.array(Data_set)
    
    output_file_np = os.path.join(directory, output_filename_np)
    data_file= fr'{directory}/info2.json'
    
    save_data = input("Would you like to save parameters? Type Y if yes: ").strip()
    if save_data[0].upper() == 'Y':
        saving = True
        while saving:
            try:
                np.savetxt(output_file_np, numpy_array, fmt='%s', delimiter=',')
                print(f"Data saved to {output_filename_np}")
                saving = False
            except IOError as e:
                print(f"Error writing to file {output_filename_np}: {e}")
                tryagin = input("Try again? Type Y if yes: ").strip()
                saving = True if tryagin.upper() == 'Y' else False
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                tryagin = input("Try again? Type Y if yes: ").strip()
                saving = True if tryagin.upper() == 'Y' else False
        
        saving2 = True        
        while saving2:
            try:
                data_to_store = {
                    "suffixes_set": suffixes_set,
                    "num_comps": num_comps,
                    "vx_filter_range": vx_filter_range,
                    "vy_filter_range": vy_filter_range,
                    "wx": wx,
                    "wy": wy,
                }
                with open(data_file, 'w') as f:  
                    json.dump(data_to_store, f, indent=4)
                print("Parameters saved to info.json")
                saving2 = False
            except IOError as e:
                print(f"Error writing to file {data_file}: {e}")
                tryagin = input("Try again? Type Y if yes: ").strip()
                saving2 = True if tryagin.upper() == 'Y' else False
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                tryagin = input("Try again? Type Y if yes: ").strip()
                saving2 = True if tryagin.upper() == 'Y' else False
        
        
    
    