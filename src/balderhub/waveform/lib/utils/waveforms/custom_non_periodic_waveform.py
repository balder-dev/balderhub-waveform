import numpy as np

from .base_non_periodic_waveform import BaseNonPeriodicWaveform


class CustomNonPeriodicWaveform(BaseNonPeriodicWaveform):
    """
    Custom individual NON-PERIODOC waveform that expects its data in the constructor
    """

    def __init__(
            self,
            data: np.ndarray,
            multiplier_amplitude_volt: float,
            delta_time_sec: float,
            offset_vdc: float = 0
    ) -> None:
        super().__init__(
            multiplier_amplitude_volt=multiplier_amplitude_volt,
            delta_time_sec=delta_time_sec,
            offset_vdc=offset_vdc,
        )
        self._data = data
        self._validate_data()

    @property
    def data(self) -> np.ndarray:
        return self._data
