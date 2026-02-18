import numpy as np

from balderhub.waveform.lib.utils.waveforms.base_periodic_waveform import BasePeriodicWaveform


class SineWaveform(BasePeriodicWaveform):
    """simple sine waveform"""

    @property
    def points_per_period(self) -> int:
        """
        :return: returns the number of points the raw-data should have
        """
        return 16384

    @property
    def data(self) -> np.ndarray:
        return np.sin(np.linspace(0, 2 * np.pi, self.points_per_period))
