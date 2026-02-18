import numpy as np

from balderhub.waveform.lib.utils.waveforms.base_periodic_waveform import BasePeriodicWaveform

class DCWaveform(BasePeriodicWaveform):
    """simple DC Waveform"""

    @property
    def data(self) -> np.ndarray:
        return np.array([0, 0])
