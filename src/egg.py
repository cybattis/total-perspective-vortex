import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class Channel:
    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.used = True


class Egg:
    def __init__(self, raw_data, view):
        self.view = view
        self.raw_data = raw_data
        self.channels: list[Channel] = self._get_channels_from_raw()
        self.sampling_rate = raw_data.info['sfreq']
        self.duration = raw_data.times[-1]
        self.index = 0  # For navigation through data segments


    def visualize(self, channel_count=10):
        """Create EEG visualization and embed it in the display frame"""
        # Create matplotlib figure
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Plot 1: Raw EEG signals (first 10 channels, first 10 seconds)
        n_channels = min(2, len(self.raw_data.ch_names))
        duration = min(10.0, self.raw_data.times[-1])

        data, times = self.raw_data[:n_channels, :int(duration * self.raw_data.info['sfreq'])]

        # Normalize and offset channels for better visualization
        data_scaled = data * 1e6  # Convert to microvolts
        offset = np.arange(n_channels) * 100
        data_offset = data_scaled + offset[:, np.newaxis]

        for i in range(n_channels):
            axes[0].plot(times[:len(data_offset[i])], data_offset[i],
                         label=self.raw_data.ch_names[i], linewidth=0.8)

        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude (µV)')
        axes[0].set_title(f'Raw EEG Signals (First {n_channels} channels, {duration}s)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot 2: Power Spectral Density
        psd_data = self.raw_data.compute_psd(fmax=50, verbose=False)
        psd_data.plot(axes=axes[1], show=False, average=True)
        axes[1].set_title('Power Spectral Density')
        axes[1].set_xlabel('Frequency (Hz)')
        axes[1].set_ylabel('Power (dB)')

        plt.tight_layout()

        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.view.display_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add dataset info
        info_text = (f"Channels: {len(self.raw_data.ch_names)}\n"
                     f"Sampling Rate: {self.raw_data.info['sfreq']} Hz\n"
                     f"Duration: {self.raw_data.times[-1]:.1f}s")
        self.view.add_info_label(info_text)


    def _get_channels_from_raw(self):
        channels = []
        for idx, ch_name in enumerate(self.raw_data.ch_names):
            ch_data, _ = self.raw_data[idx, :]
            channel = Channel(name=ch_name, data=ch_data.flatten())
            channels.append(channel)
        return channels
