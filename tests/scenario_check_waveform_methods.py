
import balder
import numpy as np

from balderhub.unit.scenarios import ScenarioUnit

from balderhub.waveform.lib.utils import waveforms


class ScenarioCheckWaveformMethods(ScenarioUnit):

    @balder.fixture('scenario')
    def sinus(self):
        return waveforms.common.SineWaveform(
            frequency_hz=5,
            amplitude_vpp=2
        )

    @balder.fixture('scenario')
    def cosinus(self):
        return waveforms.common.CosineWaveform(
            frequency_hz=5,
            amplitude_vpp=2
        )

    def test_convert_non_periodic_to_periodic(self, sinus, cosinus):

        non_periodic = waveforms.CustomNonPeriodicWaveform(
            np.sin(np.linspace(0, 10 * np.pi, 10000)),
            delta_time_sec=0.0001,
            multiplier_amplitude_volt=1
        )

        periodic_equivalent = non_periodic.get_periodic_equivalent_waveform()
        assert round(periodic_equivalent.frequency_hz, 2) == 5.00
        assert periodic_equivalent.compare(sinus, ignore_phase=True)
        assert periodic_equivalent.compare(cosinus, ignore_phase=True)

    def test_compare(self, sinus):
        assert sinus.compare(sinus), f"did not return True when comparing the same object"

    def test_compare_with_ignore_phase(self, sinus, cosinus):
        assert not sinus.compare(cosinus)
        assert sinus.compare(cosinus, ignore_phase=True)
