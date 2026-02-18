import numpy as np

from balderhub.waveform.lib.utils.waveforms.base_periodic_waveform import BasePeriodicWaveform


class CardiacWaveform(BasePeriodicWaveform):
    """simple cardiac waveform"""

    @property
    def points_per_period(self) -> int:
        """
        :return: returns the number of points the raw-data should have
        """
        return 16384

    @property
    def data(self) -> np.ndarray:
        # Phase von 0 bis fast 1 (endpoint=False → nahtlos wiederholbar)
        phase = np.linspace(0, 1, self.points_per_period, endpoint=False)

        signal = np.zeros(self.points_per_period, dtype=float)

        # P-Welle (Vorhof)
        signal += 0.25 * np.exp(-((phase - 0.18) ** 2) / (2 * 0.045 ** 2))

        # QRS complex (ventricle – the distinctive spike)
        qrs = 0.38
        signal += -0.15 * np.exp(-((phase - (qrs - 0.025)) ** 2) / (2 * 0.018 ** 2))  # Q
        signal += 1.00 * np.exp(-((phase - qrs) ** 2) / (2 * 0.012 ** 2))  # R (Hauptpeak)
        signal += -0.35 * np.exp(-((phase - (qrs + 0.035)) ** 2) / (2 * 0.022 ** 2))  # S

        # T wave (ventricular recovery)
        signal += 0.40 * np.exp(-((phase - 0.68) ** 2) / (2 * 0.085 ** 2))

        return signal
