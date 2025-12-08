import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Channel:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.used = True


class Egg:
    def __init__(self, file_path):
        self.raw_data = mne.io.read_raw_edf(file_path, preload=True)
        self.preprocess_data = self.raw_data.copy()
        self.channels: list[Channel] = self._get_channels_from_raw()
        self.sampling_rate = self.raw_data.info['sfreq']
        self.detail = self._get_details()
        self.current_start = 0
        self.window_duration = 10  # seconds

    def visualize(self):
        """
        Visualize EEG data.
        """
        # Create the plot
        fig = self.preprocess_data.plot(
            start=self.current_start,
            duration=self.window_duration,
            n_channels=len(self.raw_data.ch_names),
            scalings='auto',
            show=False  # Important: Do not show the plot directly
        )
        plt.tight_layout()
        return fig

    def visualize_psd(self):
        """
        Visualize the Power Spectral Density of the EEG data.
        """
        # Plot Power Spectral Density
        fig = self.preprocess_data.compute_psd(fmax=50).plot(average=True, show=False)
        plt.tight_layout()
        return fig

    def preprocess(self):
        """
        Preprocesses the raw EEG data by applying filters.
        """
        l_freq = 1.0
        h_freq = 20.0
        notch_freq = 60.0

        # Apply band-pass filter
        self.preprocess_data.filter(l_freq=l_freq, h_freq=h_freq, fir_design='firwin')

        # Apply notch filter
        self.preprocess_data.notch_filter(freqs=notch_freq, fir_design='firwin')

    def _get_channels_from_raw(self):
        channels = []
        for idx, ch_name in enumerate(self.raw_data.ch_names):
            ch_data, _ = self.raw_data[idx, :]
            channel = Channel(name=ch_name, data=ch_data.flatten())
            channels.append(channel)
        return channels

    def _get_details(self):
        details = (
            f"Number of Channels: {len(self.channels)}\n"
            f"Sampling Rate: {self.sampling_rate} Hz\n"
            f"Duration: {self.raw_data.duration:.2f} seconds\n"
        )
        return details

    def close(self):
        """Clean up resources if needed"""
        if self.raw_data is not None:
            self.raw_data.close()