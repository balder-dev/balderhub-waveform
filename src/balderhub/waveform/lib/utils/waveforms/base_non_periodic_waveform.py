from __future__ import annotations
from abc import ABC
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import resample, correlate, find_peaks, detrend

from balderhub.waveform.lib.utils.waveforms.base_periodic_waveform import BasePeriodicWaveform
from balderhub.waveform.lib.utils.waveforms.custom_periodic_waveform import CustomPeriodicWaveform
from balderhub.waveform.lib.utils.waveforms.abstract_waveform import AbstractWaveform


class BaseNonPeriodicWaveform(AbstractWaveform, ABC):
    """
    Base abstract class for describing a NON-PERIODIC waveform
    """

    def __init__(
            self,
            multiplier_amplitude_volt: float,
            delta_time_sec: float,
            offset_vdc: float = 0
    ):
        """

        :param multiplier_amplitude_volt: full amplitude for the whole range - the full range is
                                          [-<multiplier_amplitude_volt>;+<multiplier_amplitude_volt>] (internal data
                                          will be multiplied with this value)
        :param delta_time_sec: the delta time between two data points
        :param offset_vdc: the offset in volts the signal is moved
        """
        self._multiplier_amplitude_volt = multiplier_amplitude_volt
        self._delta_time_sec = delta_time_sec
        self._offset_vdc = offset_vdc

    @property
    def multiplier_amplitude_volt(self) -> float:
        """
        :return: multiplier the raw data needs to be multiplied to define the voltage
        """
        return self._multiplier_amplitude_volt

    @property
    def delta_time_sec(self) -> float:
        return self._delta_time_sec

    @property
    def offset_vdc(self) -> float:
        """
        :return: additional offset that will be added to the signal
        """
        return self._offset_vdc

    @property
    def total_time_sec(self) -> float:
        """
        :return: returns the total time of the signal in seconds
        """
        return (len(self.data) - 1) * self._delta_time_sec

    def get_data_in_volts(self) -> np.ndarray:
        return self.data * float(self._multiplier_amplitude_volt) + self._offset_vdc

    def get_resampled_version(self, interval_sec: float) -> BaseNonPeriodicWaveform:
        # pylint: disable-next=import-outside-toplevel
        from balderhub.waveform.lib.utils.waveforms.custom_non_periodic_waveform import CustomNonPeriodicWaveform

        new_data = resample(self.data, int(len(self.data) * float(self.delta_time_sec / interval_sec)))

        # make sure that it is in expected range
        new_data = np.round(new_data, decimals=6).astype(np.float64)

        return CustomNonPeriodicWaveform(
            data=new_data,
            multiplier_amplitude_volt=self._multiplier_amplitude_volt,
            delta_time_sec=interval_sec,
            offset_vdc=self._offset_vdc
        )

    def get_periodic_equivalent_waveform(self) -> CustomPeriodicWaveform:
        """
        This method converts the NON-PERIODIC waveform into a PERIODIC waveform. It uses correlation to determine the
        periodic pattern within the signal.

        It raises a ValueError in case it does not detect a valid periodic pattern within the data.

        :return: a new PERIOD waveform that describes the periodic pattern within this signal
        """
        # TODO rework
        self_data = self.data
        # do autocorrelation and try to find periodic signal
        signal_detrended = detrend(self_data)
        auto_corr = correlate(signal_detrended, signal_detrended, mode='full')
        auto_corr = auto_corr[len(auto_corr) // 2:]  # only use positives

        # first try to find them with very high height, then go down
        for height_percent, min_distance in ((0.8, 50), (0.6, 100), (0.4, 200)):
            peaks, _ = find_peaks(
                auto_corr,
                height=height_percent * np.max(auto_corr),
                distance=min(min_distance, len(self_data) // 10)) # TODO
            if len(peaks) > 2:
                break

        if len(peaks) < 2:
            raise ValueError('no peaks found')

        periodic_samples = peaks[1] - peaks[0]  # TODO should we consider other peaks too?

        # extract periodic part
        n_complete_cycles = len(self_data) // periodic_samples

        if n_complete_cycles < 2:
            raise ValueError("to less data to determine periodic signal")

        # Only take complete cycles and pack them into the matrix
        trimmed = self_data[:n_complete_cycles * periodic_samples]
        cycles = trimmed.reshape((n_complete_cycles, periodic_samples))

        # TODO calc the error between the different cycles (needs to be a specific value)

        # Average all cycles → “clean average cycle”
        mean_cycle = np.mean(cycles, axis=0)

        return CustomPeriodicWaveform(
            data=mean_cycle,
            frequency_hz=float(1 / (self._delta_time_sec * len(mean_cycle))),
            amplitude_vpp=float(self._multiplier_amplitude_volt * 2),
            offset_vdc=self._offset_vdc,
            phase=0
        )

    def compare(
            self,
            with_waveform: Union[BasePeriodicWaveform, BaseNonPeriodicWaveform],
            max_allowed_rmse=0.01
    ) -> bool:

        if isinstance(with_waveform, CustomPeriodicWaveform):
            # one is periodic, the other is non-periodic
            self_in_periodic = self.get_periodic_equivalent_waveform()
            return self_in_periodic.compare(with_waveform, max_allowed_rmse=max_allowed_rmse)

        # both are non-periodic -> compare each other directly
        common_delta_time = max(self.delta_time_sec, with_waveform.delta_time_sec)
        self_data = self.get_resampled_version(common_delta_time).get_data_in_volts()
        other_data = with_waveform.get_resampled_version(common_delta_time).get_data_in_volts()
        if len(self_data) != len(other_data):
            return False
        rmse = np.sqrt(np.mean((self_data - other_data)**2))
        return rmse < max_allowed_rmse

    def _prepare_matplot(self):
        plt.figure(figsize=(10, 4))  # TODO

        time_points = np.arange(0, self.total_time_sec + self._delta_time_sec, self._delta_time_sec)

        plt.plot(time_points, self.get_data_in_volts())
        plt.title(f"{self.__class__.__name__}")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [V]")
        plt.grid(True)
