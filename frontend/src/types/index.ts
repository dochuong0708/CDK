export type Modality = 'Text' | 'Image' | 'Audio' | 'Video';
export type Analysis = { analysisId: string; modality: Modality; label: string; riskScore: number; confidence: number; evidence: string[]; detector: string; input: string; createdAt: string };
export type ApiClient = { analyzeText(text: string): Promise<Analysis>; listAnalyses(): Promise<Analysis[]>; analyzeMedia(file: File, modality: Modality): Promise<Analysis> };
