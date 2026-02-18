import numpy as np

from .base_periodic_waveform import BasePeriodicWaveform


class CustomPeriodicWaveform(BasePeriodicWaveform):
    """
    Custom individual PERIODOC waveform that expects its data in the constructor
    """

    def __init__(
            self,
            data: np.ndarray,
            frequency_hz: float,
            amplitude_vpp: float,
            offset_vdc: float = 0,
            phase: float = 0
    ) -> None:
        super().__init__(
            frequency_hz=frequency_hz,
            amplitude_vpp=amplitude_vpp,
            offset_vdc=offset_vdc,
            phase=phase
        )
        self._data = data
        self._validate_data()

    @property
    def data(self) -> np.ndarray:
        return self._data
