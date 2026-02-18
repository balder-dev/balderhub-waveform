from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from typing import TypeVar

import matplotlib.pyplot as plt

import numpy as np


AbstractWaveformTypeT = TypeVar('AbstractWaveformTypeT', bound='AbstractWaveform')


class AbstractWaveform(ABC):
    """
    Abstract base class describing a waveform
    """

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """needs to return a list with values between 0 and 1"""
        raise NotImplementedError()

    @property
    @abstractmethod
    def delta_time_sec(self) -> float:
        """
        :return: delta time between two data points
        """

    def _validate_data(self):
        data = self.data
        if max(data) > 1 or min(data) < -1:
            raise ValueError(f'data values needs to be between -1 and 1, but is between {min(data)} and {max(data)}')

    @abstractmethod
    def get_data_in_volts(self) -> np.ndarray:
        """
        This method converts the raw data according to the waveform parameters (like amplitude, phase, ..) into a
        numpy array that descibes the exact volt values.
        :return: a numpy array with voltage values
        """

    @abstractmethod
    def get_resampled_version(self, interval_sec: float) -> AbstractWaveformTypeT:
        """
        This method resamples the internal raw data to ensure that the interval_sec is the exact time between every
        step.

        :param interval_sec: the interval between every single data point
        :return: a new waveform instance with the resampled data points as raw data
        """

    @abstractmethod
    def compare(self, with_waveform: AbstractWaveform, max_allowed_rmse: float = 0.01) -> bool:
        """
        This method compares to waveforms with each other. If one of this waveforms is a PERIODIC and the other one is
        a NON-PERIODIC it will automatically try to convert the NON-PERIODIC waveform to a PERIODIC waveform.

        :param with_waveform: the other waveform to compare with
        :param max_allowed_rmse: maximal allowed RMSE value to classify both waveforms as identically
        :return: True if both waveforms are identical, otherwise False
        """

    @abstractmethod
    def _prepare_matplot(self):
        pass

    def save_plot(self, filepath: pathlib.Path) -> None:
        """
        This method creates a matplotlib plot and saves it at the given filepath.
        :param filepath: the filepath to save the plot to
        """
        self._prepare_matplot()
        plt.savefig(filepath)

    def show_plot(self) -> None:
        """
        This method shows a matplotlib plot of the waveform.
        """
        self._prepare_matplot()
        plt.show()
