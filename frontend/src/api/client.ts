import type { ApiClient, Analysis } from '../types';

type AnalysisListResponse = { items: Analysis[]; total: number };

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? '';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${apiBaseUrl}${path}`, init);
  if (!response.ok) {
    throw new Error(`DeepGuard API request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export const liveClient: ApiClient = {
  async listAnalyses() {
    const response = await request<AnalysisListResponse>('/api/analyses');
    return response.items;
  },
  async analyzeText(text) {
    return request<Analysis>('/api/analyze/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
  },
  async analyzeMedia(file, modality) {
    const body = new FormData();
    body.append('file', file);
    body.append('modality', modality);
    return request<Analysis>('/api/analyze/media', { method: 'POST', body });
  },
};