import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_data_from_json(filepath):
    try:
        with open(filepath, 'r') as file:
            json_data = json.load(file)
            
        # Convert lists back to numpy arrays
        data = {header: {'Right': [np.array(arr) for arr in json_data[header]['Right']],
                         'Left': [np.array(arr) for arr in json_data[header]['Left']]}
                for header in json_data}
        
        return data
    
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{filepath}': {e}")
        return None

# Example usage
directory = r"F:\AlterG\Control\InverseKinematics"
all_data = []

try:
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            data = load_data_from_json(filepath)
            if data is not None:
                all_data.append(data)

    # Plotting
    num_rows = 8
    num_cols = 3
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 20), sharex='col', sharey='row')

    x_values = np.arange(1, 101)  # Adjust the range as per your data

    for i, data in enumerate(all_data):
        headers = list(data.keys())

        for j, header in enumerate(headers):
            right_data = data[header]['Right']
            left_data = data[header]['Left']

            # Plot 'Right' side data
            for idx, array in enumerate(right_data):
                num_rows_array = len(array)
                for k in range(num_rows_array):
                    axs[j, i].plot(x_values, array[k], label=f'Right - Array {idx} - Row {k + 1}', alpha=0.7)

            # Plot 'Left' side data
            for idx, array in enumerate(left_data):
                num_rows_array = len(array)
                for k in range(num_rows_array):
                    axs[j, i + num_cols].plot(x_values, array[k], label=f'Left - Array {idx} - Row {k + 1}', alpha=0.7)

            # Set title for each subplot
            axs[j, i].set_title(f'{header} - Right Side')
            axs[j, i + num_cols].set_title(f'{header} - Left Side')

            # Set common labels for axes
            axs[j, i].set_xlabel('% Gait Cycle')
            axs[j, i + num_cols].set_xlabel('% Gait Cycle')
            axs[j, i].set_ylabel('Angle (deg)')

            # Optionally, adjust other subplot settings like legend and limits
            # axs[j, i].legend()
            # axs[j, i + num_cols].legend()
            axs[j, i].set_xlim(1, 100)
            axs[j, i + num_cols].set_xlim(1, 100)

    # Adjust layout to prevent overlapping of subplots
    plt.tight_layout()
    plt.show()

except FileNotFoundError:
    print(f"Directory '{directory}' not found. Please provide the correct path.")
