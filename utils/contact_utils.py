import re
import numpy as np

def parse_arduino_data(filename):
    """
    Parse Arduino output text file into NumPy arrays
    
    Args:
        filename (str): Path to the text file containing Arduino data
        
    Returns:
        tuple: (time_array, channel_array) where:
            - time_array is a 1D NumPy array of shape (N_frames,)
            - channel_array is a 2D NumPy array of shape (N_frames, 16)
    """
    # Lists to store parsed data
    times = []
    channels = []
    
    # Regular expression to extract data
    # This pattern matches "Time: X" and any number of "ChY: Z" patterns
    pattern = r'Time: (\d+)(?:, Ch(\d+): (\d+)){16}'
    
    n_lines = []
    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Extract time value using regex
            time_match = re.search(r'Time: (\d+)', line)
            if time_match:
                time_value = int(time_match.group(1))
                times.append(time_value)
                
                # Extract all channel values for this line
                channel_values = []
                for ch_num in range(16):
                    ch_pattern = f'Ch{ch_num}: (\d+)'
                    ch_match = re.search(ch_pattern, line)
                    if ch_match:
                        channel_values.append(int(ch_match.group(1)))
                    else:
                        # If channel not found, add 0 (or could raise an error)
                        channel_values.append(0)
                
                channels.append(channel_values)
    
    # Convert lists to NumPy arrays
    time_array = np.array(times)  # Shape: (N_frames,)
    channel_array = np.array(channels)  # Shape: (N_frames, 16)

    # import matplotlib.pyplot as plt
    # plt.plot(time_array)
    # plt.show()
    
    return time_array, channel_array