# Profibus-DP telegram analysis
# Input is segment data from signal analysis
# Output is analyzed data of the segments
from tabulate import tabulate
import numpy as np


# Class to store segment data
class Segment:
    def __init__(
        self, ID, length, start_time, end_time, state, n_bits, bin_form):
        self.ID = ID
        self.length = length
        self.start_time = start_time
        self.end_time = end_time
        self.state = state
        self.n_bits = n_bits
        self.bin_form = bin_form


# Class to store Profibus frame
class Frame:
    def __init__(self, start_bit, data_bits, hex_form, parity_bit, stop_bit):
        self.start_bit = start_bit
        self.data_bits = data_bits
        self.hex_form = hex_form
        self.parity_bit = parity_bit
        self.stop_bit = stop_bit
        self.frame_status = False
    
    def frame_check(self):
        # Start bit always 0
        start_status = False
        if self.start_bit == "0":
            start_status = True
        
        # Parity bit check
        parity_status = False
        if self.data_bits.count("1") % 2 == 0:
            if self.parity_bit == "0":
                parity_status = True
        else:
            if self.parity_bit == "1":
                parity_status = True

        # Start bit always 1
        stop_status = False
        if self.stop_bit == "1":
            stop_status = True

        if start_status and parity_status and stop_status:
            self.frame_status = True


# Collect the segments
def save_segments(BIT_FRAME):
    segments = []
    print("Save telegram segments.")
    n = 1 # Segment number starting from 1
    
    while input("Continue [y/n]? ") == "y":
        print("{}. segment".format(n))
        length = int(input("Segment length (integer): "))
        start_time = float(input("Segment start time: "))
        end_time = float(input("Segment end time: "))
        state = input("Segment state (0/1): ")
        n_bits = get_bit_length(length, BIT_FRAME)
        bin_form = get_bin_form(n_bits, state)
        new_segment = Segment(
            n, length, start_time, end_time, state, n_bits, bin_form)
        segments.append(new_segment)
        n += 1
        print("Segment saved.")
    
    # Save segments
    if input("Save to file [y/n]? ") == "y":
        filename = input("Give filename: ")
        segments = np.array(segments)
        np.save(filename, segments)
    return segments


# Load existing segment array
def load_segments(filename):
    segments = np.load(filename, allow_pickle=True)
    return segments


# Calculate the number of bits in the segment
def get_bit_length(length, BIT_FRAME):
    n_bits = round(length / BIT_FRAME)
    return n_bits


# Form the binary representation of the segment
def get_bin_form(n_bits, state):
    bin_form = state * n_bits
    return bin_form


def separate_frames(telegram):
    # Check leading ones if exist
    i = 0
    while telegram[i] == '1':
        telegram = telegram[i+1:]
    
    # Slice the telegram to frames
    frames = []
    i = 0
    while i < len(telegram):
        raw_frame = telegram[i:i+11]
        new_frame = Frame(
            raw_frame[0], # start bit
            raw_frame[1:8][::-1], # data bits
            hex(int(raw_frame[1:8][::-1], 2)), # hex form
            raw_frame[9], # parity bit
            raw_frame[10] # stop bit
        )
        
        # Check and add to array
        new_frame.frame_check()
        frames.append(new_frame)
        i += 11 # Next frame
    
    print("\nFrames:")
    HEADER = ["ST", "Octet", "Hex", "PB", "SP", "Frame status"]
    data = []
    for frame in frames:
        row = [
            frame.start_bit, 
            frame.data_bits, 
            frame.hex_form,
            frame.parity_bit, 
            frame.stop_bit, 
            frame.frame_status
        ]
        data.append(row)
    print(tabulate(data, HEADER, tablefmt="simple"))


# Print the saved segments to table
def print_segments(segments):
    HEADER = ["Segment", "Length", "Start", "End", "No. bits", "Binary"]
    data = []
    for segment in segments:
        row = [
            segment.ID, 
            segment.length, 
            segment.start_time, 
            segment.end_time, 
            segment.n_bits, 
            segment.bin_form
            ]
        data.append(row)
    print(tabulate(data, HEADER, tablefmt="simple"))


# Print the data in bytes
def print_data(segments):
    # Combine all data
    telegram = ""
    for segment in segments:
        telegram = telegram + segment.bin_form
    print("\nTelegram data:", telegram)
    print("Data length: ", len(telegram), "b")
    separate_frames(telegram)


def main():
    print("### PROFIBUS ANALYSIS ###")
    # Load existing segments
    if input("Load existing segments [y/n]: ") == "y":
        filename = input("Give filename: ")
        segments = load_segments(filename)
    # Save segments
    elif input("Save segments [y/n]: ") == "y":
        print("Start information for analysis.")
        BAUD_RATE = int(input("Give the baud rate (Mbps): "))
        BIT_FRAME = int((1 / (BAUD_RATE * 1000000)) * 1000000000)
        print("Start info ready, using settings:")
        print("- Baud rate: ", BAUD_RATE, "Mbps")
        print("- Transfer time / bit: ", BIT_FRAME, "ns")
        segments = save_segments(BIT_FRAME)
    
    # Print saved segments
    print_segments(segments)

    # Print the whole telegram message
    print_data(segments)


if __name__ == '__main__':
    main()