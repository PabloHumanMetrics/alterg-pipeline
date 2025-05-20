#!/usr/bin/env python
# coding: utf-8

# my_script.py

# **Imports**
import os
import re
import pickle
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
from scipy import stats

# **DataLoader Class**
class DataLoader:
    @staticmethod
    def load_mot_file(file_path, header_line=8):
        data = pd.read_csv(file_path, sep='\t', header=header_line)
        return data

    @staticmethod
    def load_hdf5_file(file_path):
        return h5py.File(file_path, 'r')

    @staticmethod
    def get_mot_files(folder_path):
        return list((folder_path / 'IKResults').glob("*.mot"))

    @staticmethod
    def extract_trial_name(file_path, pattern=r"\\([^\\]+)_ik"):
        match = re.search(pattern, str(file_path))
        if match:
            return match.group(1)
        else:
            # Try matching with forward slashes for Unix-based paths
            pattern_unix = r"/([^/]+)_ik"
            match_unix = re.search(pattern_unix, str(file_path))
            if match_unix:
                return match_unix.group(1)
        return None

# **Utils Class**
class Utils:
    @staticmethod
    def get_ROI(file, trial_name):
        print('ROI = ', file['NexusData'][trial_name]['Info']['Region'][:])
        return file['NexusData'][trial_name]['Info']['Region'][:]

    @staticmethod
    def get_sampling_rate(file, trial_name):
        import numpy as np
        trial_data = file['NexusData'][trial_name]

        # Check if 'HeelType' exists in trial_data
        if 'HeelType' in trial_data:
            heel_type_data = trial_data['HeelType'][()]
            if isinstance(heel_type_data, bytes):
                heel_type = heel_type_data.decode('utf-8')
            elif isinstance(heel_type_data, np.ndarray):
                heel_type = ''.join(chr(code) for code in heel_type_data.flatten())
            else:
                heel_type = str(heel_type_data)
            print(f"HeelType: {heel_type}")

            if heel_type == 'IMUs':
                devices = trial_data['devices']
                # Find the first IMU device starting with 'TS0'
                imu_devices = [name for name in devices if name.startswith('TS0')]
                if not imu_devices:
                    print('No IMU devices found starting with TS0. Using default sampling rate of 225 Hz.')
                    return 225  # Default value
                imu_device = imu_devices[0]
                imu_device_data = devices[imu_device]

                if 'accel_x' in imu_device_data and 'Rate' in imu_device_data['accel_x']:
                    sampling_rate = imu_device_data['accel_x']['Rate'][()]
                    sampling_rate = np.array(sampling_rate).item()
                    print('IMU heels Sampling =', sampling_rate)
                    return sampling_rate
                else:
                    print(f"Could not find 'Rate' under 'accel_x' in IMU device '{imu_device}'. Using default sampling rate of 225 Hz.")
                    return 225  # Default value

            elif heel_type == 'Kinematics':
                print('Kinematic Heels = 100 Hz conversion')
                return 100
            else:
                print('Unknown HeelType:', heel_type)
                print('No definition of heel type; defaulting to 225 Hz conversion')
                return 225
        else:
            # 'HeelType' does not exist in trial_data
            print("HeelType does not exist in trial_data. Using default sampling rate of 225 Hz.")
            return 225  # Default sampling rate

    @staticmethod
    def filter_events_within_roi(events, ROI, event_sampling_rate):
        """
        Filters event indices to include only those within the ROI, handling the conversion
        between the ROI sampling rate (100 Hz) and the event sampling rate.
        """
        # Extract start and end frames from ROI
        roi_start_frame_100hz = ROI[0][0]
        roi_end_frame_100hz = ROI[1][0]

        # Convert ROI frames to time (seconds) using 100 Hz sampling rate
        roi_start_time = roi_start_frame_100hz / 100.0
        roi_end_time = roi_end_frame_100hz / 100.0

        # Ensure events is a 1D NumPy array of scalars
        events = np.array(events).flatten()

        # Convert event indices to time (seconds) using event_sampling_rate
        event_times = events / event_sampling_rate

        # Filter event times within ROI times using a boolean mask
        mask = (event_times >= roi_start_time) & (event_times <= roi_end_time)
        filtered_event_times = event_times[mask]

        # Convert filtered event times back to indices at event_sampling_rate
        filtered_events = np.round(filtered_event_times * event_sampling_rate).astype(int)

        # Convert to list
        filtered_events = np.array(filtered_events).flatten().tolist()

        return filtered_events

    @staticmethod
    def indices_to_time(indices, sampling_rate, frame_offset=0):
        indices = np.array(indices).flatten()
        indices = indices.tolist()
        times = [(idx - frame_offset) / sampling_rate for idx in indices]
        return times

    @staticmethod
    def process_side_strides(ipsi_heels, ipsi_toes, contra_heels, contra_toes, header, data):
        """
        Processes strides for one side and collects associated events.
        """
        def get_indices_from_times(event_times, data_times):
            indices = []
            for t in event_times:
                idx = np.argmin(np.abs(data_times - t))
                indices.append(idx)
            return indices

        strideList = []

        # Extract data times
        data_times = data['time'].values

        # Convert event times to indices
        ipsi_heel_indices = get_indices_from_times(ipsi_heels, data_times)
        ipsi_toe_indices = get_indices_from_times(ipsi_toes, data_times)
        contra_heel_indices = get_indices_from_times(contra_heels, data_times)
        contra_toe_indices = get_indices_from_times(contra_toes, data_times)

        for idx in range(len(ipsi_heel_indices) - 1):
            start_idx = ipsi_heel_indices[idx]
            end_idx = ipsi_heel_indices[idx + 1]
            start_time = data_times[start_idx]
            end_time = data_times[end_idx]

            # Extract stride data between start_idx and end_idx
            stride_data = data.iloc[start_idx:end_idx][header].values
            stride_time = data.iloc[start_idx:end_idx]['time'].values

            if len(stride_data) > 0:
                pass
            else:
                print('No stride data found for this stride.')
                continue  # Skip to the next stride if there's no data

            events_in_stride = []
            all_events = []
            for t in ipsi_toes:
                if start_time <= t <= end_time:
                    all_events.append({'time': t, 'type': 'ipsi_toe_off'})
            # Contralateral events
            for t in contra_toes:
                if start_time <= t <= end_time:
                    all_events.append({'time': t, 'type': 'contra_toe_off'})
            for t in contra_heels:
                if start_time <= t <= end_time:
                    all_events.append({'time': t, 'type': 'contra_heel_strike'})

            # Sort events by time to maintain sequence
            all_events.sort(key=lambda x: x['time'])

            # Calculate event indices relative to stride_data
            for event in all_events:
                relative_time = (event['time'] - start_time) / (end_time - start_time)
                event_idx = int(relative_time * (len(stride_data) - 1))
                event['index'] = event_idx
                events_in_stride.append(event)

            # Correct orientation if necessary
            stride_data = Utils.correct_orientation(stride_data)

            # Store stride data and events
            stride_info = {
                'stride_data': stride_data,
                'start_time': start_time,
                'end_time': end_time,
                'events': events_in_stride
            }

            strideList.append(stride_info)

        return strideList

    @staticmethod
    def correct_orientation(data_array):
        if np.mean(data_array) < 0:  # Assuming that correct data has positive mean
            return -data_array
        return data_array

    @staticmethod
    def normalize_strides(strides_list, normalization_type='cycle', target_length=100):
        normalized_strides_list = []

        for stride_info in strides_list:
            stride_data = stride_info['stride_data']
            stride_length = len(stride_data)

            if normalization_type == 'cycle':
                # Normalize stride data to target_length using interpolation
                stride_data_normalized = np.interp(
                    np.linspace(0, stride_length - 1, target_length),
                    np.arange(stride_length),
                    stride_data
                )

                # Adjust event indices
                adjusted_events = []
                for event in stride_info['events']:
                    original_index = event['index']
                    normalized_index = int((original_index / (stride_length - 1)) * (target_length - 1))
                    event_adjusted = event.copy()
                    event_adjusted['index'] = normalized_index
                    adjusted_events.append(event_adjusted)

                # Update stride_info with normalized data and adjusted events
                stride_info_normalized = stride_info.copy()
                stride_info_normalized['normalized_stride_data'] = stride_data_normalized
                stride_info_normalized['normalized_events'] = adjusted_events

            elif normalization_type == 'phase':
                # Placeholder for 'phase' normalization (to be developed)
                stride_info_normalized = stride_info  # No changes for now

            else:
                raise ValueError("Invalid normalization_type. Choose 'cycle' or 'phase'.")

            normalized_strides_list.append(stride_info_normalized)

        return normalized_strides_list

    @staticmethod
    def apply_conventions(strides_r, strides_l, header):
        """
        Applies the required processing to the strides for the given header.
        """
        # Headers to skip
        skip_headers = [
            'time', 'pelvis_tilt', 'pelvis_list', 'pelvis_tx', 'pelvis_ty', 'pelvis_tz',
            'hip_rotation_r', 'hip_rotation_l', 'subtalar_angle_l', 'subtalar_angle_r',
            'mtp_angle_l', 'mtp_angle_r'
        ]

        if header in skip_headers:
            # Copy normalized data to processed data without changes
            if strides_r is not None:
                for stride_info in strides_r:
                    stride_info['processed_stride_data'] = stride_info['normalized_stride_data']
            if strides_l is not None:
                for stride_info in strides_l:
                    stride_info['processed_stride_data'] = stride_info['normalized_stride_data']
            return strides_r, strides_l

        # Headers where right strides need to be multiplied by -1
        headers_flip_right = ['pelvis_rotation', 'lumbar_bending', 'lumbar_rotation']

        # Headers where both sides need to be multiplied by -1
        headers_multiply_minus1 = ['lumbar_extension']

        # Headers where we need to check for flipped strides
        headers_check_flip = ['hip_flexion', 'knee_angle', 'ankle_angle']

        # Process right strides
        if strides_r is not None:
            # Collect all normalized stride data for right side
            all_stride_data_r = [stride_info['normalized_stride_data'] for stride_info in strides_r]

            for stride_info in strides_r:
                # Start with a copy of the normalized data
                stride_data = stride_info['normalized_stride_data'].copy()

                # Apply conventions
                if header in headers_flip_right:
                    stride_data = -stride_data

                if header in headers_multiply_minus1:
                    stride_data = -stride_data

                if any(h in header for h in headers_check_flip) or header == 'hip_adduction_r':
                    # Correct flipped strides using correlation with mean stride
                    stride_data = Utils.correct_flipped_stride(stride_data, all_stride_data_r)

                # Demean if necessary
                if header == 'pelvis_rotation':
                    mean_value = np.mean(stride_data)
                    stride_data -= mean_value

                # Store the processed data
                stride_info['processed_stride_data'] = stride_data

        # Process left strides similarly
        if strides_l is not None:
            # Collect all normalized stride data for left side
            all_stride_data_l = [stride_info['normalized_stride_data'] for stride_info in strides_l]

            for stride_info in strides_l:
                # Start with a copy of the normalized data
                stride_data = stride_info['normalized_stride_data'].copy()

                # Apply conventions
                if header in headers_multiply_minus1:
                    stride_data = -stride_data

                if any(h in header for h in headers_check_flip) or header == 'hip_adduction_l':
                    # Correct flipped strides using correlation with mean stride
                    stride_data = Utils.correct_flipped_stride(stride_data, all_stride_data_l)

                # Demean if necessary
                if header == 'pelvis_rotation':
                    mean_value = np.mean(stride_data)
                    stride_data -= mean_value

                # Store the processed data
                stride_info['processed_stride_data'] = stride_data

        return strides_r, strides_l

    @staticmethod
    def correct_flipped_stride(stride_data, all_stride_data_list):
        """
        Identifies and corrects a single stride if it is flipped, based on correlation with the mean stride.
        """
        # Compute the mean stride from all strides
        all_strides_array = np.stack(all_stride_data_list, axis=0)
        mean_stride = np.mean(all_strides_array, axis=0)

        # Compute the correlation coefficient between the stride and the mean stride
        correlation = np.corrcoef(stride_data, mean_stride)[0, 1]

        # If the correlation is negative, flip the stride
        if correlation < 0:
            corrected_stride_data = -stride_data
        else:
            corrected_stride_data = stride_data

        return corrected_stride_data

    @staticmethod
    def compute_mean_and_variability(strides_list, variability='SD'):
        """
        Computes the mean stride and variability (SD, SE, or 95% CI) across a list of strides.
        """
        if not strides_list:
            return None, None, None

        # Collect all processed stride data
        stride_data_list = [stride_info['processed_stride_data'] for stride_info in strides_list]
        # Stack stride data into a 2D array (strides x data points)
        stride_data_array = np.stack(stride_data_list, axis=0)
        # Compute the mean across strides
        mean_stride = np.mean(stride_data_array, axis=0)

        # Compute variability
        if variability == 'SD':
            var_stride = np.std(stride_data_array, axis=0)
            lower_bound = mean_stride - var_stride
            upper_bound = mean_stride + var_stride
        elif variability == 'SE':
            var_stride = np.std(stride_data_array, axis=0) / np.sqrt(stride_data_array.shape[0])
            lower_bound = mean_stride - var_stride
            upper_bound = mean_stride + var_stride
        elif variability == 'CI':
            from scipy import stats
            confidence = 0.95
            n = stride_data_array.shape[0]
            stderr = stats.sem(stride_data_array, axis=0)
            t_value = stats.t.ppf((1 + confidence) / 2., n - 1)
            margin_of_error = t_value * stderr
            lower_bound = mean_stride - margin_of_error
            upper_bound = mean_stride + margin_of_error
        else:
            raise ValueError("Invalid variability type. Choose 'SD', 'SE', or 'CI'.")

        return mean_stride, lower_bound, upper_bound

    @staticmethod
    def plot_mean_and_variability(mean_stride, lower_bound, upper_bound, header, side_label, color):
        """
        Plots the mean stride and variability area.
        """
        if mean_stride is None:
            return

        x_values = np.linspace(0, 100, len(mean_stride))  # Stride percentage

        plt.plot(x_values, mean_stride, color=color, label=f'{side_label} Mean')
        plt.fill_between(x_values, lower_bound, upper_bound, color=color, alpha=0.3, label=f'{side_label} Variability')
        plt.xlabel('Stride Percentage (%)')
        plt.ylabel(header)
        plt.legend()

    @staticmethod
    def save_participant_strides(stride_data, participant_id, participant_folderpath):
        """
        Saves the participant's concatenated strides into the Kinematics folder.
        """
        # Access the participant's data from stride_data
        participant_data = stride_data

        # Initialize a dictionary to store concatenated strides across trials
        participant_concatenated_strides = {}

        # Iterate over trials and concatenate strides
        for trial_name, trial_processed_strides in participant_data.items():
            for header, stride_data in trial_processed_strides.items():
                # Initialize the header in participant_concatenated_strides if not already present
                if header not in participant_concatenated_strides:
                    participant_concatenated_strides[header] = {
                        'strides_r': [],
                        'strides_l': [],
                        'header_type': stride_data['header_type']
                    }

                # Extend the participant's strides with the strides from this trial
                if stride_data['strides_r']:
                    participant_concatenated_strides[header]['strides_r'].extend(stride_data['strides_r'])
                if stride_data['strides_l']:
                    participant_concatenated_strides[header]['strides_l'].extend(stride_data['strides_l'])

        # Define the path to the Kinematics folder
        kinematics_folder = participant_folderpath / 'Kinematics'

        # Create the Kinematics folder if it doesn't exist
        if not kinematics_folder.exists():
            os.makedirs(kinematics_folder)
            print(f"Created Kinematics folder at {kinematics_folder}")

        # Define the file path to save the strides
        strides_file_path = kinematics_folder / 'concatenated_strides.pkl'

        # Save the participant_concatenated_strides dictionary to the file
        with open(strides_file_path, 'wb') as f:
            pickle.dump(participant_concatenated_strides, f)

        print(f"Saved concatenated strides to {strides_file_path}")

# **Trial Class**
class Trial:
    def __init__(self, trial_name, mot_file_path, heel_file_path):
        self.trial_name = trial_name
        self.mot_file_path = mot_file_path
        self.heel_file_path = heel_file_path
        self.data = None
        self.heels_r = []
        self.heels_l = []
        self.toes_r = []
        self.toes_l = []
        self.roi = None
        self.sampling_rate = None
        self.stride_data = {}
        self.load_data()

    def load_data(self):
        self.data = DataLoader.load_mot_file(self.mot_file_path)
        with DataLoader.load_hdf5_file(self.heel_file_path) as file:
            if self.trial_name in file['NexusData']:
                self.roi = Utils.get_ROI(file, self.trial_name)
                self.sampling_rate = Utils.get_sampling_rate(file, self.trial_name)
                trial_data = file['NexusData'][self.trial_name]
                self.heels_r = trial_data['Right']['heels'][:]
                self.heels_l = trial_data['Left']['heels'][:]
                self.toes_r = trial_data['Right']['Toes'][:]
                self.toes_l = trial_data['Left']['Toes'][:]
            else:
                print(f"Trial {self.trial_name} not found in HDF5 file.")
                # Set default values or handle appropriately
                self.roi = [[0], [0]]  # Default ROI
                self.sampling_rate = 100  # Default sampling rate

    def process_trial(self):
        self.filter_events()
        self.convert_indices_to_times()
        self.process_strides()
        # Optionally, you can add plotting here using the Plotter class

    def filter_events(self):
        self.heels_r = Utils.filter_events_within_roi(self.heels_r, self.roi, self.sampling_rate)
        self.heels_l = Utils.filter_events_within_roi(self.heels_l, self.roi, self.sampling_rate)
        self.toes_r = Utils.filter_events_within_roi(self.toes_r, self.roi, self.sampling_rate)
        self.toes_l = Utils.filter_events_within_roi(self.toes_l, self.roi, self.sampling_rate)

    def convert_indices_to_times(self):
        self.heels_r_times = Utils.indices_to_time(self.heels_r, self.sampling_rate)
        self.heels_l_times = Utils.indices_to_time(self.heels_l, self.sampling_rate)
        self.toes_r_times = Utils.indices_to_time(self.toes_r, self.sampling_rate)
        self.toes_l_times = Utils.indices_to_time(self.toes_l, self.sampling_rate)

    def process_strides(self):
        stride_processor = StrideProcessor(self.data, self.heels_r_times, self.heels_l_times,
                                           self.toes_r_times, self.toes_l_times)
        self.stride_data = stride_processor.process_all_headers()

    def save_processed_data(self, participant_folderpath):
        Utils.save_participant_strides({self.trial_name: self.stride_data}, self.trial_name, participant_folderpath)

# **StrideProcessor Class**
class StrideProcessor:
    def __init__(self, data, heels_r_times, heels_l_times, toes_r_times, toes_l_times):
        self.data = data
        self.heels_r_times = heels_r_times
        self.heels_l_times = heels_l_times
        self.toes_r_times = toes_r_times
        self.toes_l_times = toes_l_times
        self.processed_strides = {}

    def process_all_headers(self):
        bilateral_headers, right_only_headers, left_only_headers = self.categorize_headers()
        self.process_bilateral_headers(bilateral_headers)
        self.process_right_only_headers(right_only_headers)
        self.process_left_only_headers(left_only_headers)
        return self.processed_strides

    def categorize_headers(self):
        bilateral_headers = []
        right_only_headers = []
        left_only_headers = []

        for header in self.data.columns:
            if header.endswith('_r'):
                right_only_headers.append(header)
            elif header.endswith('_l'):
                left_only_headers.append(header)
            else:
                bilateral_headers.append(header)

        return bilateral_headers, right_only_headers, left_only_headers

    def process_bilateral_headers(self, headers):
        for header in headers:
            print(f'Processing bilateral header: {header}')
            strides_r = Utils.process_side_strides(
                self.heels_r_times, self.toes_r_times,
                self.heels_l_times, self.toes_l_times,
                header, self.data
            )
            strides_l = Utils.process_side_strides(
                self.heels_l_times, self.toes_l_times,
                self.heels_r_times, self.toes_r_times,
                header, self.data
            )
            strides_r = Utils.normalize_strides(strides_r)
            strides_l = Utils.normalize_strides(strides_l)
            strides_r, strides_l = Utils.apply_conventions(strides_r, strides_l, header)
            self.processed_strides[header] = {
                'strides_r': strides_r,
                'strides_l': strides_l,
                'header_type': 'bilateral'
            }

    def process_right_only_headers(self, headers):
        for header in headers:
            print(f'Processing right-only header: {header}')
            strides_r = Utils.process_side_strides(
                self.heels_r_times, self.toes_r_times,
                self.heels_l_times, self.toes_l_times,
                header, self.data
            )
            strides_r = Utils.normalize_strides(strides_r)
            strides_r, _ = Utils.apply_conventions(strides_r, None, header)
            self.processed_strides[header] = {
                'strides_r': strides_r,
                'strides_l': None,
                'header_type': 'right_only'
            }

    def process_left_only_headers(self, headers):
        for header in headers:
            print(f'Processing left-only header: {header}')
            strides_l = Utils.process_side_strides(
                self.heels_l_times, self.toes_l_times,
                self.heels_r_times, self.toes_r_times,
                header, self.data
            )
            strides_l = Utils.normalize_strides(strides_l)
            _, strides_l = Utils.apply_conventions(None, strides_l, header)
            self.processed_strides[header] = {
                'strides_r': None,
                'strides_l': strides_l,
                'header_type': 'left_only'
            }

# **Participant Class**
class Participant:
    def __init__(self, participant_id, data_folder, heel_path_template):
        self.participant_id = participant_id
        self.data_folder = Path(data_folder) / f"{participant_id}"
        self.heel_path = heel_path_template.format(f"{participant_id}")
        self.trials = {}
        self.load_trials()

    def load_trials(self):
        mot_files = DataLoader.get_mot_files(self.data_folder)
        for file_path in mot_files:
            trial_name = DataLoader.extract_trial_name(file_path)
            if trial_name:
                trial = Trial(trial_name, file_path, self.heel_path)
                self.trials[trial_name] = trial

    def process_trials(self):
        for trial in self.trials.values():
            trial.process_trial()

    def save_processed_data(self):
        participant_stride_data = {}
        for trial_name, trial in self.trials.items():
            participant_stride_data[trial_name] = trial.stride_data
        Utils.save_participant_strides(participant_stride_data, self.participant_id, self.data_folder)

# **Main Script**
def main():
    # File paths and configurations
    folderpath = r'F:\AlterG\Control\Data'
    heel_path_template = r'F:\AlterG\Control\Data\{}\Gait\DevicesData.mat'
    participant_range = range(1, 22)  # Modify as needed

    for trial in participant_range:
        participant_id = f"{trial:02d}"
        print(f'\nProcessing Participant ID: {participant_id}')
        participant = Participant(participant_id, folderpath, heel_path_template)
        participant.process_trials()
        participant.save_processed_data()

if __name__ == "__main__":
    main()
