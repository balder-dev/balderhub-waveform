from __future__ import annotations
from abc import ABC
from typing import Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.core import correlate
from scipy.signal import resample

from balderhub.waveform.lib.utils.waveforms.abstract_waveform import AbstractWaveform

if TYPE_CHECKING:
    from balderhub.waveform.lib.utils.waveforms import BaseNonPeriodicWaveform


class BasePeriodicWaveform(AbstractWaveform, ABC):
    """
    Base abstract class for describing a PERIODIC waveform
    """
    def __init__(
            self,
            frequency_hz: float,
            amplitude_vpp: float,
            offset_vdc: float = 0,
            phase: float = 0
    ):
        self._frequency_hz = frequency_hz
        self._amplitude_vpp = amplitude_vpp
        self._offset_vdc = offset_vdc
        self._phase = phase % (2*np.pi)

    @property
    def frequency_hz(self) -> float:
        """
        :return: the frequency that should be applied to this waveform
        """
        return self._frequency_hz

    @property
    def delta_time_sec(self) -> float:
        return (1 / self.frequency_hz) / len(self.data)

    @property
    def amplitude_vpp(self) -> float:
        """
        :return: the amplitude value the raw data will be multiplied with
        """
        return self._amplitude_vpp

    @property
    def offset_vdc(self) -> float:
        """
        :return: the signal offset the raw data will be added to
        """
        return self._offset_vdc

    @property
    def phase(self) -> float:
        """
        :return: the phase of the signal
        """
        return self._phase

    def get_data_in_volts(self) -> np.ndarray:
        split_at = int(len(self.data) * (self.phase % (2 * np.pi)) / (2 * np.pi))
        phase_cleaned_raw_data = np.concatenate((self.data[split_at:], self.data[:split_at]))

        return phase_cleaned_raw_data * float(self._amplitude_vpp) / 2 - self._offset_vdc

    def get_resampled_version(self, interval_sec: float) -> BasePeriodicWaveform:
        # pylint: disable-next=import-outside-toplevel
        from balderhub.waveform.lib.utils.waveforms.custom_periodic_waveform import CustomPeriodicWaveform

        new_data_point_cnt = int(len(self.data) * float(self.delta_time_sec / interval_sec))
        if new_data_point_cnt == 0:
            raise ValueError('can not resample data, because no data point would be there with the new interval time')
        new_data = resample(self.data, new_data_point_cnt)

        # make sure that it is in expected range
        new_data = np.round(new_data, decimals=6).astype(np.float64)

        return CustomPeriodicWaveform(
            data=new_data,
            frequency_hz=self.frequency_hz,
            amplitude_vpp=self.amplitude_vpp,
            offset_vdc=self.offset_vdc,
            phase=self.phase
        )

    def get_phase_difference_to(
            self,
            other_periodic_waveform: BasePeriodicWaveform,
            max_allowed_rmse: float = 0.01,
    ) -> Union[float, None]:
        """
        This method gets the phase difference between two periodic waveforms.

        :param other_periodic_waveform: the other waveform
        :return: the phase between [0,2*pi] from this waveform to the other waveform or None, if both waveforms are
                 different
        """
        # pylint: disable-next=import-outside-toplevel
        from balderhub.waveform.lib.utils.waveforms import CustomPeriodicWaveform

        if self.amplitude_vpp != other_periodic_waveform.amplitude_vpp:
            return False
        if self.frequency_hz != other_periodic_waveform.frequency_hz:
            return False
        common_delta_time = max(self.delta_time_sec, other_periodic_waveform.delta_time_sec)
        self_signal = self.get_resampled_version(common_delta_time).get_data_in_volts()
        other_signal = other_periodic_waveform.get_resampled_version(common_delta_time).get_data_in_volts()

        # double one of the curves
        self_signal = np.tile(self_signal, 2)
        # determine the position with the best correlation
        auto_corr = correlate(self_signal, other_signal, mode='valid')
        best_pos = np.argmax(auto_corr)
        self_signal = self_signal[best_pos:best_pos + len(self.data)]
        if len(self_signal) < len(other_signal):
            return False
        phase = 2 * np.pi * (best_pos/len(self.data))
        new_modified_waveform = CustomPeriodicWaveform(
            data=other_periodic_waveform.data,
            frequency_hz=other_periodic_waveform.frequency_hz,
            amplitude_vpp=other_periodic_waveform.amplitude_vpp,
            offset_vdc=other_periodic_waveform.offset_vdc,
            phase=other_periodic_waveform.phase - phase
        )

        if self.compare(new_modified_waveform, max_allowed_rmse, ignore_phase=False):
            return phase
        return None

    def compare(
            self,
            with_waveform: Union[BasePeriodicWaveform, BaseNonPeriodicWaveform],
            max_allowed_rmse: float = 0.01,
            ignore_phase: bool = False
    ) -> bool:
        """
        This method compares to waveforms with each other. If one of the waveforms is a PERIODIC and the other one is
        a NON-PERIODIC it will automatically try to convert the NON-PERIODIC waveform to a PERIODIC waveform.

        If ``ignore_phase`` is True, it will ignore the position of the waveform and attempts to find the same
        waveform by shifting it. For example, in a cosine and sine comparison where ``ignore_phase=True``, True
        would also be returned in the compare, since the signals are identical but with a phase shift of 90 degree.

        :param with_waveform: the other waveform to compare with
        :param max_allowed_rmse: maximal allowed RMSE value to classify both waveforms as identically
        :return: True if both waveforms are identical, otherwise False
        """
        # TODO improve
        # pylint: disable-next=import-outside-toplevel
        from balderhub.waveform.lib.utils.waveforms.custom_non_periodic_waveform import CustomNonPeriodicWaveform

        if isinstance(with_waveform, CustomNonPeriodicWaveform):
            # one is periodic, the other is non-periodic
            with_waveform_periodic = with_waveform.get_periodic_equivalent_waveform()
            return with_waveform_periodic.compare(
                with_waveform,
                max_allowed_rmse=max_allowed_rmse,
                ignore_phase=ignore_phase
            )
        # have the same type -> compare each other directly
        if self.frequency_hz != with_waveform.frequency_hz:
            return False

        if ignore_phase:
            return self.get_phase_difference_to(with_waveform, max_allowed_rmse=max_allowed_rmse) is not None

        common_delta_time = max(self.delta_time_sec, with_waveform.delta_time_sec)
        self_signal = self.get_resampled_version(common_delta_time).get_data_in_volts()
        other_signal = with_waveform.get_resampled_version(common_delta_time).get_data_in_volts()

        rmse = np.sqrt(np.mean((self_signal - other_signal) ** 2))
        return rmse < max_allowed_rmse

    def _prepare_matplot(self):
        # TODO
        plt.figure(figsize=(10, 4))  # TODO

        delta_time_sec = (1 / self.frequency_hz) / len(self.data)

        time_points = np.arange(0, 1 / self.frequency_hz, delta_time_sec)

        plt.plot(time_points, self.get_data_in_volts())
        plt.title(f"{self.__class__.__name__}")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [V]")
        plt.grid(True)
