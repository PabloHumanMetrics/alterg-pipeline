# Import packages
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path
import pandas as pd
import h5py
import re
import os 
import json 

# File paths
folderpath = Path(r'F:\AlterG\Control\Data')
HeelPathTemplate = r'F:\AlterG\Control\Data\{}\Gait\DevicesData.mat'
header_line = 8
pattern = r"\\([^\\]+)_ik"
base_json_path = r"F:\AlterG\Control\InverseKinematics"

# Dictionaries
concatenated_stride_data = {}
all_strides_dict = {'Right': {}, 'Left': {}}

#Function to interpolate strides to a common length and concatenate them
def interpolate_strides(strides, target_length=100):
    interpolated_values_list = []
    x_new = np.linspace(0, 1, target_length)
    
    for stride_array in strides:
        for stride in stride_array:
            if len(stride) < 1:
                continue
            
            x_old = np.linspace(0, 1, len(stride))
            
            if len(stride) < 4:
                interp_func = interp1d(x_old, stride, kind='linear', bounds_error=False, fill_value="extrapolate")
            else:
                interp_func = interp1d(x_old, stride, kind='cubic', bounds_error=False, fill_value="extrapolate")
            
            interpolated_values = interp_func(x_new)
            interpolated_values_list.append(interpolated_values)
    
    return np.array(interpolated_values_list)

# Function to plot all strides per header in a single plot
def plot_stride_list(strideList, side, header):
    """    
    Parameters:
    strideList (list): List containing the stride data.
    side (str): Side of the stride ('Right' or 'Left').
    header (str): The header to plot.

    """
    plt.figure(figsize=(12, 6))
    for stride in strideList:
        plt.plot(stride, label=f'{side} Side', alpha=0.6)
    
    plt.title(f'All Strides for {header} ({side} Side)')
    plt.xlabel('Sample Points')
    plt.ylabel('Values')
    #plt.legend()
    plt.show()
    plt.close()

# Function to plot all interpolated strides per header in a single plot.
def plot_interpolated_values(interpolated_values, side, header):
    """
    Parameters:
    interpolated_values (ndarray): Array containing the interpolated stride data.
    side (str): Side of the stride ('Right' or 'Left').
    header (str): The header to plot.

    """
    plt.figure(figsize=(12, 6))
    for stride in interpolated_values:
        plt.plot(stride, alpha=0.6, label=f'{side} Side' if side not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.title(f'Interpolated Strides for {header} ({side} Side)')
    plt.xlabel('Sample Points')
    plt.ylabel('Values')
    #plt.legend()
    plt.show()

# Function to plot concatenated strides for each header and side
def plot_subject(concatenated_stride_data, side, header):
    plt.figure(figsize=(12, 6))
    for stride_list in concatenated_stride_data[header][side]:
        for stride in stride_list:
            plt.plot(stride, alpha=0.6, label=f'{side} Side' if side not in plt.gca().get_legend_handles_labels()[1] else "")
    
    plt.title(f'Individual Strides for {header} ({side} Side) for {trial}')
    plt.xlabel('% Gait Cycle')
    plt.ylabel(f'{header} (deg)')
    #plt.legend()
    plt.show()
    plt.close()

# Function to plot mean and standard deviation for each header and side
def plot_mean_std(concatenated_stride_data):
    mean_strides_dict = {}
    std_strides_dict = {}

    for side in ['Right', 'Left']:
        mean_strides_dict[side] = {}
        std_strides_dict[side] = {}

        for header in concatenated_stride_data:
            all_strides_array = np.concatenate(concatenated_stride_data[header][side])
            mean_strides_dict[side][header] = np.mean(all_strides_array, axis=0)
            std_strides_dict[side][header] = np.std(all_strides_array, axis=0)

    # Plot mean with standard deviation shaded region for each header and side
    plot_handles = []
    for side in ['Right', 'Left']:
        for header in mean_strides_dict[side]:
            plt.figure(figsize=(12, 6))
            plot_handle, = plt.plot(mean_strides_dict[side][header], label=f'{side} Side')
            plt.fill_between(np.arange(len(mean_strides_dict[side][header])),
                             mean_strides_dict[side][header] - std_strides_dict[side][header],
                             mean_strides_dict[side][header] + std_strides_dict[side][header],
                             alpha=0.3)
            plt.title(f'{header} ({side} Side) for {trial}')
            plt.xlabel('% Gait Cycle')
            plt.ylabel(f'{header} (deg)')
            #plt.legend()
            plot_handles.append(plot_handle)
            plt.show()
            plt.close()

    return plot_handles

# Calculate length of each stride and then their mean and standard deviation
def clean_strides_mean_std(stride_list):
    lengths = [len(stride) for stride in stride_list]
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    # Calculate the minimum acceptable length
    min_length = mean_length - std_length
    # Filter out strides that are shorter than the minimum acceptable length
    cleaned_strides = [stride for stride in stride_list if len(stride) >= min_length]
    
    return cleaned_strides

# Calculate the interquartile range
def clean_strides_iqr(stride_list):
    lengths = [len(stride) for stride in stride_list]
    
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = np.percentile(lengths, 25)
    Q3 = np.percentile(lengths, 75)
    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    
    # Calculate the lower bound for acceptable lengths
    lower_bound = Q1 - 1.5 * IQR
    # Filter out strides that are shorter than the lower bound
    cleaned_strides = [stride for stride in stride_list if len(stride) >= lower_bound]
    
    return cleaned_strides

# Save the data to json files
def save_data_to_json(data, filepath):
    json_data = {header: {'Right': [arr.tolist() for arr in data[header]['Right']],
                          'Left': [arr.tolist() for arr in data[header]['Left']]}
                 for header in data}

    with open(filepath, 'w') as file:
        json.dump(json_data, file)
    #print(f"Data saved to {filepath}")

# Processing of each mot file
for trial in range(1, 22):
    current_folderpath = folderpath / f"{trial:02d}"
    current_HeelPath = HeelPathTemplate.format(f"{trial:02d}")
    mot_files = list((current_folderpath / 'IKResults').glob("*.mot"))
    json_save_path = os.path.join(base_json_path, f"Subject{trial:02d}.json")
    
    strideList_dict = {}  
    
    for file_path in mot_files:
        file_path_str = str(file_path)
        data = pd.read_csv(file_path, sep='\t', header=header_line)
        current_headers = data.columns

        # Extract trial name from file path
        match = re.search(pattern, file_path_str)
        if match:
            trial = match.group(1)

            # Check if trial exists in the HDF5 file
            with h5py.File(current_HeelPath, 'r') as file:
                if trial in file['NexusData']:
                    heels_r = []
                    heels_l = []

                    nexus_data_r = file['NexusData'][trial]['Right']['emg_idx'][:]
                    nexus_data_l = file['NexusData'][trial]['Left']['emg_idx'][:]

                    # Extend heels_r and heels_l with converted data
                    heels_r.extend([array.item() / 10 for array in nexus_data_r])
                    heels_l.extend([array.item() / 10 for array in nexus_data_l])
                    # print ('R Heels: ', heels_r)
                    # print ('L Heels: ', heels_l)
                    #print(f'Lengths corroborated: heels_r = {len(heels_r)}, heels_l = {len(heels_l)}')

                    # Process stride data for each header
                    for header in data.columns:
                        specific_header = 'hip_flexion_r'
                        strideList_r = []
                        strideList_l = []
                        cleaned_strides_mean_std_r = []
                        cleaned_strides_mean_std_l = []
                        data_between_elements = []
                        
                        # Figure out direction & depending on header +> do something with data.values*** 









   

                        # Process Right side
                        for idx in range(len(heels_r) - 1):
                            start_index = int(heels_r[idx])
                            end_index = int(heels_r[idx + 1])
                            #print(f"start_index (heels_r[{idx}]): {start_index}, end_index (heels_r[{idx + 1}]): {end_index}")
                            data_between_elements = data.iloc[start_index:end_index][header].values
                            if data_between_elements.size > 0:
                                strideList_r.append(data_between_elements)
                        
                        # Print and visualise the structure of strideList_r
                        #print(f"Structure of strideList_r for header '{header}':", strideList_r)
                        #plot_stride_list(strideList_r, 'Right', header)

                        # Process Left side
                        for idx in range(len(heels_l) - 1):
                            start_index = int(heels_l[idx])
                            end_index = int(heels_l[idx + 1])
                            #print(f"start_index (heels_l[{idx}]): {start_index}, end_index (heels_l[{idx + 1}]): {end_index}")
                            data_between_elements = data.iloc[start_index:end_index][header].values
                            if data_between_elements.size > 0:
                                strideList_l.append(data_between_elements)
                        
                        # Print and visualise the structure of strideList_l
                        #print(f"Structure of strideList_l for header '{header}':", strideList_l)
                        #plot_stride_list(strideList_l, 'Left', header)
                        
                        # Apply the cleaning methods
                        cleaned_strides_mean_std_r = clean_strides_mean_std(strideList_r)
                        cleaned_strides_mean_std_l = clean_strides_mean_std(strideList_l)
                        
                        # Plot the cleaned strides using mean and std method
                        #plot_stride_list(cleaned_strides_mean_std_r, 'Right', header)
                        #plot_stride_list(cleaned_strides_mean_std_l, 'Left', header)
                        
                        # Add stride lists to strideList_dict
                        if header not in strideList_dict:
                            strideList_dict[header] = {'Right': [], 'Left': []}

                        #if header == specific_header:
                            #print(f"Before appending to strideList_dict, cleaned_strides_mean_std_r shape: {[np.array(s).shape for s in cleaned_strides_mean_std_r]}")
                        #print(f"Before appending to strideList_dict, cleaned_strides_mean_std_l shape: {[np.array(s).shape for s in cleaned_strides_mean_std_l]}")

                        strideList_dict[header]['Right'].append(cleaned_strides_mean_std_r)
                        strideList_dict[header]['Left'].append(cleaned_strides_mean_std_l)
            
                        # Interpolate stride data to 100 points
                        interpolated_values_right = interpolate_strides(strideList_dict[header]['Right'])
                        interpolated_values_left = interpolate_strides(strideList_dict[header]['Left'])

                        # if header == specific_header:
                        #     print(f"After interpolation, interpolated_values_right shape: {interpolated_values_right.shape}")
                            #print(f"After interpolation, interpolated_values_left shape: {interpolated_values_left.shape}")

                        # Visualize the interpolated_values_right and interpolated_values_left
                        # plot_interpolated_values(interpolated_values_right, 'Right', header)
                        # plot_interpolated_values(interpolated_values_left, 'Left', header)

                        # Store interpolated values
                        if header not in concatenated_stride_data:
                            concatenated_stride_data[header] = {'Right': [], 'Left': []}
                        concatenated_stride_data[header]['Right'].append(interpolated_values_right)
                        concatenated_stride_data[header]['Left'].append(interpolated_values_left)

                        # Print the shape of concatenated_stride_data for debugging
                        # if header == specific_header:
                        #     print(f"Trial: {trial}, Header: {header}, Right strides count: {len(concatenated_stride_data[header]['Right'])}")
                        #     for i, arr in enumerate(concatenated_stride_data[header]['Right']):
                        #         print(f"  Stride {i} shape: {arr.shape}")
    
    for header in concatenated_stride_data:
        plot_subject(concatenated_stride_data, 'Right', header)
        plot_subject(concatenated_stride_data, 'Left', header)
    #plot_handles = plot_mean_std(concatenated_stride_data)
    # save_data_to_json(concatenated_stride_data, json_save_path)






    # def extract_and_plot_heel_y(data):
    # # Extract the Y columns for RHeel, LHeel, RHip, and LHip markers
    #     time = data['Time']
    #     rheel_y = data['RHeel_Y']
    #     lheel_y = data['LHeel_Y']
    #     rhip_y = data['RHip_Y']
    #     lhip_y = data['LHip_Y']
        
    #     # Plot the Y trajectories for RHeel and LHeel markers against time
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(time, lheel_y, label='LHeel_Y')
    #     plt.plot(time, rheel_y, label='RHeel_Y')
    #     plt.title('LHeel and RHeel Y Trajectories against Time')
    #     plt.xlabel('Time')
    #     plt.ylabel('Y Coordinate')
    #     plt.legend()
    #     plt.show()
    
    #     return time, lheel_y, rheel_y, lhip_y, rhip_y