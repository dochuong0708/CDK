"""Week 6 numerical/contract tests. Passing tests is not model accuracy."""
import json

import numpy as np
import pytest
import soundfile as sf
import torch

from core.tools.audio_features_tool import audio_features_tool, decode_audio
from core.tools.audio_anomaly_tool import AudioAnomalyTool
from app.detectors.audio.model import AudioCNN
from app.detectors.audio.pipeline import CONFIG


def tone(path, frequency=440, rate=16000, seconds=1, amplitude=.2):
    y = amplitude*np.sin(2*np.pi*frequency*np.arange(int(rate*seconds))/rate)
    sf.write(path, y, rate, subtype="FLOAT")
    return path


def test_frequency_axis_waveform_and_resampling(tmp_path):
    result = audio_features_tool(tone(tmp_path/"stereo.wav", rate=48000))
    spectrogram = result["spectrogram"]
    band = int(np.argmax(np.mean(spectrogram["valuesDb"], axis=1)))
    assert abs(spectrogram["frequencyHz"][band]-440) < 100
    assert result["sampleRate"] == 16000 and result["originalSampleRate"] == 48000
    waveform = result["waveform"]
    assert max(waveform["maximum"]) == pytest.approx(.2, abs=.01)
    assert min(waveform["minimum"]) == pytest.approx(-.2, abs=.01)
    assert 0.98 < waveform["timeSeconds"][-1] <= 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("rate,seconds,amplitude", [(4000, 1, .2), (16000, .1, .2), (16000, 61, .2), (16000, 1, 0)])
def test_reject_invalid_signal_constraints(tmp_path, rate, seconds, amplitude):
    with pytest.raises(ValueError):
        decode_audio(tone(tmp_path/"invalid.wav", rate=rate, seconds=seconds, amplitude=amplitude))


def test_frequency_flags_do_not_invent_model_prediction(tmp_path):
    tool = AudioAnomalyTool(str(tmp_path/"missing.pt"))
    low = tool(tone(tmp_path/"low.wav"), explain=False)
    high = tool(tone(tmp_path/"high.wav", frequency=7000, amplitude=1.3), explain=False)
    assert "high_energy_above_6kHz" not in low["signalFlags"]
    assert set(high["signalFlags"]) == {"clipping", "high_energy_above_6kHz"}
    assert high["fakeScore"] is None and high["status"] == "unavailable"
    assert high["frequencyDiagnostics"]["validatedForSpoofDetection"] is False


def test_class_order_threshold_and_checkpoint_validation(tmp_path):
    torch.set_num_threads(2)
    model = AudioCNN()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.classifier.bias[1] = 2
    path = tmp_path/"model.pt"
    checkpoint = dict(architecture="audio-cnn-v1", config=CONFIG, classes=["real", "fake"],
                      threshold=.5, state_dict=model.state_dict())
    torch.save(checkpoint, path)
    sample = tone(tmp_path/"tone.wav")
    result = AudioAnomalyTool(str(path))(sample, explain=False)
    assert result["fakeScore"] == pytest.approx(1/(1+np.exp(-2)), abs=1e-6)
    assert result["predictedClass"] == "fake"
    checkpoint["threshold"] = .95
    torch.save(checkpoint, path)
    assert AudioAnomalyTool(str(path))(sample, explain=False)["predictedClass"] == "real"
    checkpoint["classes"] = ["fake", "real"]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="Checkpoint"):
        AudioAnomalyTool(str(path))(sample, explain=False)
