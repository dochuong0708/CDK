export type Modality = 'Text' | 'Image' | 'Audio' | 'Video';
export type AudioDetails = { fakeScore: number; threshold: number; explainedWindow: number; xaiPngBase64: string; windows: {start: number; end: number; fakeScore: number}[] };
export type Analysis = { analysisId: string; modality: Modality; label: string; riskScore: number; confidence: number; evidence: string[]; detector: string; input: string; createdAt: string; audio?: AudioDetails | null };
export type ApiClient = { analyzeText(text: string): Promise<Analysis>; listAnalyses(): Promise<Analysis[]>; analyzeMedia(file: File, modality: Modality): Promise<Analysis> };
