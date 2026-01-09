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
    def __init__(self, file_path, settings):
        self.raw_data = mne.io.read_raw_edf(file_path, preload=True)
        self.raw_data = mne.add_reference_channels(self.raw_data, ref_channels=["EEG 999"])
        self.preprocess_data = self.raw_data.copy()
        self.channels: list[Channel] = self._get_channels_from_raw()
        self.sampling_rate = self.raw_data.info['sfreq']
        self.current_start = 0
        self.window_duration = 10  # seconds
        self.notch_filter = settings.get("notch_filter", None)
        self.low_freq = 5
        self.cutoff_freq = settings.get("cutoff_value", 100)


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

    def plot_sensor(self):
        """
        Visualize EEG data.
        """
        # Create the plot
        fig = self.preprocess_data.plot_sensors(ch_type="eeg")
        plt.tight_layout()
        return fig

    def visualize_psd(self):
        """
        Visualize the Power Spectral Density of the EEG data.
        """
        # Plot Power Spectral Density
        fig = self.preprocess_data.compute_psd(fmax=50).plot(average=True, picks="data", exclude="bads", amplitude=False)
        plt.tight_layout()
        return fig

    def preprocess(self):
        """
        Preprocesses the raw EEG data by applying filters.
        """
        print("Preprocessing EEG data...")
        print(f"Applying band-pass filter: {self.low_freq} - {self.cutoff_freq} Hz")
        print(f"Applying notch filter at: {self.notch_filter} Hz")

        # Apply band-pass filter
        self.preprocess_data.filter(l_freq=self.low_freq, h_freq=self.cutoff_freq, fir_design='firwin')

        # Apply notch filter
        self.preprocess_data.notch_filter(freqs=self.notch_filter, fir_design='firwin')

    def _get_channels_from_raw(self):
        channels = []
        for idx, ch_name in enumerate(self.raw_data.ch_names):
            ch_data, _ = self.raw_data[idx, :]
            channel = Channel(name=ch_name, data=ch_data.flatten())
            channels.append(channel)
        return channels

    def get_details(self):
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