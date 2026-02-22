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
        return self.data * float(self._multiplier_amplitude_volt) - self._offset_vdc

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
        if len(self.data) < 300:
            raise ValueError("Not enough samples to reliably detect periodicity.")

        # Remove linear trend
        detrended = detrend(self.data)

        # Normalized autocorrelation (positive lags only)
        auto_corr = correlate(detrended, detrended, mode='full')
        auto_corr = auto_corr[len(auto_corr) // 2:]
        auto_corr = auto_corr / (auto_corr[0] + 1e-12)  # normalize

        # Adaptive minimum lag (prevents picking noise as period)
        min_lag = max(8, len(self.data) // 80)

        # Find candidate periods
        peaks, _ = find_peaks(
            auto_corr[min_lag:],
            height=0.28,  # a bit higher than before
            prominence=0.18,
            distance=min_lag // 3
        )

        if len(peaks) == 0:
            raise ValueError("No significant periodic component detected.")

        candidate_periods = np.sort(peaks + min_lag)  # <-- sort smallest first!

        # Evaluate from shortest to longest
        best_period = None
        best_rel_var = np.inf
        best_n_cycles = 0

        for period in candidate_periods[:12]:  # limit to first 12 promising ones
            n_cycles = len(self.data) // period
            if n_cycles < 3:
                continue

            trimmed = self.data[:n_cycles * period]
            cycles = trimmed.reshape((n_cycles, period))

            # Relative variation across cycles (very robust metric)
            cycle_std_mean = np.mean(np.std(cycles, axis=0))
            p2p = np.ptp(np.mean(cycles, axis=0)) + 1e-9
            rel_var = cycle_std_mean / p2p

            # 22% is a sweet spot for real oscilloscope data (noise + small jitter)
            if rel_var < 0.22:
                # Found a good short period → take it immediately!
                best_period = period
                best_rel_var = rel_var
                best_n_cycles = n_cycles
                break

            # Keep track of the overall best in case none is below threshold
            if rel_var < best_rel_var:
                best_rel_var = rel_var
                best_period = period
                best_n_cycles = n_cycles

        if best_period is None:
            raise ValueError("Could not find a consistent periodic pattern.")

        # Build the clean average cycle
        trimmed = self.data[:best_n_cycles * best_period]
        cycles = trimmed.reshape((best_n_cycles, best_period))
        mean_cycle = np.mean(cycles, axis=0)

        return CustomPeriodicWaveform(
            data=mean_cycle,
            frequency_hz=float(1 / (self._delta_time_sec * best_period)),
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
