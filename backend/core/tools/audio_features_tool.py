"""Bounded, JSON-safe feature extraction using librosa (not model preprocessing)."""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

MAX_BYTES = 20 * 1024 * 1024
EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def decode_audio(path: Path, target_rate=16000):
    if path.suffix.lower() not in EXTENSIONS or path.stat().st_size > MAX_BYTES:
        raise ValueError("Chỉ nhận WAV/MP3/FLAC/OGG tối đa 20 MB.")
    try:
        with sf.SoundFile(path) as source:
            rate, channels, frames = source.samplerate, source.channels, source.frames
            if not 8000 <= rate <= 192000 or not 1 <= channels <= 8:
                raise ValueError("Sample rate hoặc số kênh không được hỗ trợ.")
            if not .5 <= frames / rate <= 60:
                raise ValueError("Âm thanh phải dài từ 0.5 đến 60 giây.")
            # Bound the decoded allocation even if a malformed header understates length.
            audio = source.read(frames=rate * 60 + 1, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError) as exc:
        raise ValueError("Không giải mã được âm thanh; hãy xuất lại WAV hoặc MP3.") from exc
    if len(audio) > rate * 60 or not np.isfinite(audio).all():
        raise ValueError("Dữ liệu âm thanh không hợp lệ.")
    mono = audio.mean(axis=1)
    if np.max(np.abs(mono), initial=0) < 1e-6:
        raise ValueError("Âm thanh im lặng hoặc các kênh triệt tiêu nhau.")
    metadata = dict(originalSampleRate=rate, originalChannels=channels, duration=len(mono)/rate,
                    clippingFraction=float(np.mean(np.abs(audio) >= .999)))
    y = librosa.resample(mono, orig_sr=rate, target_sr=target_rate) if target_rate and rate != target_rate else mono
    return y, metadata


def audio_features_tool(path: Path) -> dict:
    y, metadata = decode_audio(path)
    power = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=512, hop_length=160, n_mels=80)
    db = librosa.power_to_db(power, ref=np.max, top_db=80)
    mfcc = librosa.feature.mfcc(S=db, n_mfcc=20)
    stride = max(1, int(np.ceil(len(y)/1000)))
    blocks = [y[i:i+stride] for i in range(0, len(y), stride)]
    step = max(1, int(np.ceil(db.shape[1]/300)))
    spectrum = np.abs(librosa.stft(y, n_fft=512, hop_length=160)) ** 2
    high = float(spectrum[192:].sum() / max(float(spectrum.sum()), 1e-12))
    return dict(status="ok", sampleRate=16000, **metadata,
                mfcc=dict(count=20, mean=mfcc.mean(axis=1).tolist(), std=mfcc.std(axis=1).tolist()),
                waveform=dict(timeSeconds=(np.arange(len(blocks))*stride/16000).tolist(),
                              minimum=[float(b.min()) for b in blocks], maximum=[float(b.max()) for b in blocks]),
                spectrogram=dict(valuesDb=db[:, ::step].tolist(),
                                 timeSeconds=(np.arange(0, db.shape[1], step)*.01).tolist(),
                                 frequencyHz=librosa.mel_frequencies(n_mels=82, fmin=0, fmax=8000)[1:-1].tolist()),
                signalMetrics=dict(highFrequencyEnergyRatio=high, rms=float(np.sqrt(np.mean(y*y)))),
                note="MFCC và phổ mô tả tín hiệu; bất thường phổ không tự chứng minh deepfake. Phổ này dùng librosa, khác preprocessing của CNN.")
