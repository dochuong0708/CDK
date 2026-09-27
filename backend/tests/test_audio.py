import base64
from io import BytesIO

import numpy as np
import pytest
import soundfile as sf
import torch
from PIL import Image

from app.detectors.audio.model import AudioCNN, grad_cam
from app.detectors.audio.pipeline import CONFIG, load_audio, segments, spectrogram
from app.detectors.audio.inference import load_checkpoint, predict
from audio_cli import manifest, metrics

torch.set_num_threads(2)


@pytest.fixture
def tone(tmp_path):
    path = tmp_path / 'tone.wav'
    t = np.arange(22050 * 5) / 22050
    sf.write(path, np.column_stack([.5*np.sin(2*np.pi*440*t)]*2), 22050)
    return path


def test_resample_segment_and_spectrum(tone):
    y = load_audio(tone)
    assert len(y) == 80000
    clips = list(segments(y))
    assert [(s, d) for s, d, _ in clips] == [(0, 4), (4, 1)]
    x = spectrogram(clips[0][2])
    assert x.shape == (1, 80, 401)
    assert torch.isfinite(x).all() and 0 <= x.min() <= x.max() <= 1


def test_invalid_audio(tmp_path):
    path = tmp_path / 'bad.wav'
    path.write_bytes(b'not audio')
    with pytest.raises(ValueError):
        load_audio(path)
    sf.write(path, np.zeros(16000), 16000)
    with pytest.raises(ValueError, match='im lặng'):
        load_audio(path)
    sf.write(path, np.ones(1600), 16000)
    with pytest.raises(ValueError, match='60'):
        load_audio(path)


def test_cam_is_class_dependent_and_zero_safe():
    model = AudioCNN().eval()
    with torch.no_grad():
        for parameter in model.features.parameters():
            parameter.fill_(.01)
        model.classifier.weight[0].fill_(-1)
        model.classifier.weight[1].fill_(1)
    x = torch.ones(1, 1, 80, 401)
    _, positive = grad_cam(model, x, 1)
    _, negative = grad_cam(model, x, 0)
    assert positive.shape == (80, 401) and positive.max() == pytest.approx(1)
    assert negative.max() == 0 and np.isfinite(negative).all()


def test_checkpoint_predict_plot_and_weighting(tone, tmp_path):
    model = AudioCNN()
    checkpoint = tmp_path / 'model.pt'
    torch.save(dict(architecture='audio-cnn-v1', config=CONFIG, classes=['real', 'fake'],
                    threshold=.5, state_dict=model.state_dict()), checkpoint)
    loaded, threshold = load_checkpoint(checkpoint)
    result = predict(tone, loaded, threshold)
    assert result['fakeScore'] == pytest.approx(sum(w['fakeScore']*(w['end']-w['start']) for w in result['windows'])/5)
    assert result['explainedWindow'] == int(np.argmax([w['fakeScore'] for w in result['windows']]))
    image = Image.open(BytesIO(base64.b64decode(result['xaiPngBase64'])))
    assert image.width >= 1000 and image.height >= 800
    assert result['predictedClass'] in ['real', 'fake']


def test_metrics_and_leakage(tone, tmp_path):
    report = metrics(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), .5)
    assert report['roc_auc'] == 1 and report['eer_approx'] == 0
    path = tmp_path / 'manifest.csv'
    other = tmp_path / 'other.wav'
    sf.write(other, np.ones(16000)*.1, 16000)
    path.write_text('path,label,speaker,split\ntone.wav,real,same,train\nother.wav,fake,same,dev\n')
    with pytest.raises(ValueError, match='nhiều split'):
        manifest(path)


def test_audio_api_missing_model_and_upload_cleanup(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    monkeypatch.setattr(main, 'upload_dir', tmp_path)
    monkeypatch.setenv('DEEPGUARD_AUDIO_MODEL', str(tmp_path / 'absent.pt'))
    client = TestClient(main.app)
    response = client.post('/api/analyze/media', data={'modality': 'Audio'},
                           files={'file': ('../../escape.wav', b'invalid', 'audio/wav')})
    assert response.status_code == 503
    assert list(tmp_path.iterdir()) == []


def test_audio_api_and_storage(tone, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as main
    from app.db.store import AnalysisStore
    from app.detectors.audio.audio_detector import AudioDetector
    checkpoint = tmp_path / 'model.pt'
    torch.save(dict(architecture='audio-cnn-v1', config=CONFIG, classes=['real', 'fake'],
                    threshold=.5, state_dict=AudioCNN().state_dict()), checkpoint)
    monkeypatch.setenv('DEEPGUARD_AUDIO_MODEL', str(checkpoint))
    monkeypatch.setattr(main.orchestrator, 'audio_detector', AudioDetector())
    monkeypatch.setattr(main, 'store', AnalysisStore(str(tmp_path / 'test.db')))
    client = TestClient(main.app)
    for _ in range(2):
        response = client.post('/api/analyze/media', data={'modality': 'Audio'},
                               files={'file': ('tone.wav', tone.read_bytes(), 'audio/wav')})
        assert response.status_code == 200, response.text
        body = response.json()
        saved = client.get('/api/analyses/' + body['analysisId']).json()
        assert saved['audio']['xaiPngBase64'] == body['audio']['xaiPngBase64']
    response = client.post('/api/analyze/media', data={'modality': 'Audio'},
                           files={'file': ('bad.wav', b'invalid', 'audio/wav')})
    assert response.status_code == 422
