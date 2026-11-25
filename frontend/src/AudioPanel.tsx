import { useState } from 'react';
import { api } from './api';
import type { Analysis, AudioXai } from './types';

type HeatmapKind = 'spectrogram' | 'attribution';

function heatColor(value: number, scale: number, kind: HeatmapKind) {
  const amount = Math.max(0, Math.min(1, kind === 'attribution' ? Math.abs(value) / Math.max(scale, 1e-8) : value));
  if (kind === 'attribution') {
    const target = value >= 0 ? [190, 45, 54] : [42, 105, 190];
    const rgb = target.map(channel => Math.round(255 + (channel - 255) * amount));
    return `rgb(${rgb.join(',')})`;
  }
  const rgb = [28 + 196 * amount, 58 + 150 * amount, 102 + 140 * amount].map(Math.round);
  return `rgb(${rgb.join(',')})`;
}

function AudioHeatmap({ values, scale, kind, label }: {
  values: number[][];
  scale: number;
  kind: HeatmapKind;
  label: string;
}) {
  const rows = values.length;
  const columns = values[0]?.length ?? 0;
  if (!rows || !columns) return <p>Không có dữ liệu bản đồ.</p>;
  return <svg className="audio-heatmap" viewBox={`0 0 ${columns} ${rows}`} role="img" aria-label={label}>
    {values.flatMap((row, rowIndex) => row.map((value, columnIndex) =>
      <rect key={`${rowIndex}-${columnIndex}`} x={columnIndex} y={rows - rowIndex - 1}
        width="1.02" height="1.02" fill={heatColor(value, scale, kind)}>
        <title>{`Dải ${rowIndex + 1}, khung ${columnIndex + 1}: ${value.toFixed(3)}`}</title>
      </rect>))}
  </svg>;
}

function AudioExplanation({ xai }: { xai: AudioXai }) {
  return <div className="audio-xai-results">
    <figure>
      <figcaption>Phổ log filter-bank · trục ngang: thời gian, trục dọc: dải tần</figcaption>
      <AudioHeatmap values={xai.spectrogram} scale={1} kind="spectrogram" label="Phổ log filter-bank theo thời gian và dải tần" />
    </figure>
    <figure>
      <figcaption>Occlusion · đỏ: vùng ủng hộ spoof; xanh: vùng làm điểm spoof tăng khi bị che</figcaption>
      <AudioHeatmap values={xai.attribution} scale={xai.attributionScale} kind="attribution" label="Bản đồ attribution occlusion theo thời gian và dải tần" />
    </figure>
    <p className="audio-xai-note">{xai.note} {xai.attributionMeaning}</p>
  </div>;
}

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
    <p className="eyebrow">AUDIO AI · LFCC + EXTRA TREES + OCCLUSION XAI</p>
    <h2 id="audio-title">Phân loại giọng nói thật / giả</h2>
    <p>Tải giọng nói WAV, FLAC hoặc OGG, từ 0,5 đến 60 giây, tối đa 20 MB. Cần model Extra Trees đã được huấn luyện trên máy chủ.</p>
    <label htmlFor="audio-file">Chọn tệp âm thanh</label>
    <input id="audio-file" type="file" accept=".wav,.flac,.ogg" disabled={busy}
      onChange={e => { setFile(e.target.files?.[0]); setResult(undefined); setError(''); }} />
    <button className="button primary" disabled={!file || busy} onClick={analyze}>
      {busy ? 'Đang phân tích và tạo XAI…' : 'Phân tích giọng nói'}
    </button>
    {error && <p role="alert" className="validation">{error}</p>}
    {result && <div aria-live="polite">
      <h3>{result.label}</h3>
      <p>Điểm spoof: <b>{((result.audioScore ?? result.riskScore / 100) * 100).toFixed(1)}%</b>
        {result.audioThreshold != null && <> · Ngưỡng: {(result.audioThreshold * 100).toFixed(1)}%</>}
        {result.audioLatencyMs != null && <> · Thời gian: {result.audioLatencyMs.toFixed(1)} ms</>}</p>
      <p>Điểm Extra Trees chưa được hiệu chuẩn thành xác suất. Kết quả là sàng lọc tham khảo, không tự khẳng định nguồn gốc giọng nói.</p>
      {result.audioXai && <AudioExplanation xai={result.audioXai} />}
      {!result.audioXai && <p>Phản hồi không có dữ liệu XAI âm thanh.</p>}
    </div>}
  </section>;
}
