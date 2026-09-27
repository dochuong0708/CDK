import { useState } from 'react';
import { api } from './api';
import type { Analysis } from './types';

export function AudioPanel() {
  const [file, setFile] = useState<File>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState<Analysis>();
  async function analyze() {
    if (!file) return;
    setBusy(true); setError(''); setResult(undefined);
    try { setResult(await api.analyzeMedia(file, 'Audio')); }
    catch (e) { setError(e instanceof Error ? e.message : 'Không thể phân tích âm thanh.'); }
    finally { setBusy(false); }
  }
  return <section id="audio-workspace" className="panel audio-panel" aria-labelledby="audio-title">
    <p className="eyebrow">AUDIO AI · MEL-SPECTROGRAM + GRAD-CAM</p>
    <h2 id="audio-title">Phân loại giọng nói thật / giả</h2>
    <p>Tải giọng nói WAV, FLAC hoặc OGG, từ 0,5 đến 60 giây, tối đa 20 MB. Cần mô hình đã huấn luyện trên máy chủ.</p>
    <label htmlFor="audio-file">Chọn tệp âm thanh</label>
    <input id="audio-file" type="file" accept=".wav,.flac,.ogg" disabled={busy}
      onChange={e => { setFile(e.target.files?.[0]); setResult(undefined); setError(''); }} />
    <button className="button primary" disabled={!file || busy} onClick={analyze}>
      {busy ? 'Đang phân tích và tạo XAI…' : 'Phân tích giọng nói'}
    </button>
    {error && <p role="alert" className="validation">{error}</p>}
    {result?.audio && <div aria-live="polite">
      <h3>{result.label}</h3>
      <p>Điểm lớp giả: <b>{(result.audio.fakeScore * 100).toFixed(1)}%</b> · Ngưỡng: {(result.audio.threshold * 100).toFixed(1)}%</p>
      <p>Đây là điểm mô hình chưa hiệu chỉnh, không phải xác suất xác thực. Grad-CAM giải thích lớp “giả” tại cửa sổ có điểm giả cao nhất, không xác định chắc chắn vị trí giả mạo.</p>
      <img className="audio-xai" src={`data:image/png;base64,${result.audio.xaiPngBase64}`}
        alt="Waveform toàn file, phổ Mel và Grad-CAM của cửa sổ có điểm giả cao nhất" />
      <a download="deepguard-audio-xai.png" href={`data:image/png;base64,${result.audio.xaiPngBase64}`}>Tải đồ thị XAI</a>
      <table><thead><tr><th>Cửa sổ (giây)</th><th>Điểm lớp giả</th><th>Giải thích</th></tr></thead>
        <tbody>{result.audio.windows.map((window, index) => <tr key={window.start}>
          <td>{window.start.toFixed(2)} – {window.end.toFixed(2)}</td>
          <td>{(window.fakeScore * 100).toFixed(1)}%</td>
          <td>{index === result.audio?.explainedWindow ? 'Được hiển thị trong Grad-CAM' : ''}</td>
        </tr>)}</tbody></table>
    </div>}
  </section>;
}
