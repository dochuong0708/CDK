export type AppMode = 'text' | 'image' | 'audio'

export type RiskLabel = 'SAFE' | 'SUSPICIOUS' | 'DANGEROUS'

export type ClaimVerdict =
  | 'supported'
  | 'contradicted'
  | 'mixed'
  | 'insufficient_evidence'

export interface SuspiciousPhrase {
  phrase: string
  start: number
  end: number
  category: string
  reason: string
}

export interface FactCheckEvidence {
  title: string
  url: string
  snippet: string
  source: string
  published_at?: string
  stance: 'supports' | 'contradicts' | 'context' | 'irrelevant'
  priority_publisher: boolean
  similarity?: number | null
}

export type FactualityLabel = RiskLabel | 'UNVERIFIED'

export interface ClaimItem {
  claim: string
  search_query?: string
  verdict: ClaimVerdict
  confidence: number
  explanation: string
  evidence: FactCheckEvidence[]
}

export interface ParagraphAnalysis {
  index: number
  text: string
  risk_score: number
  label: RiskLabel
  model_score: number | null
  rule_score: number
  style_risk: number
  style_label: RiskLabel
  fraud_risk: number
  factuality_risk: number | null
  factuality_label: FactualityLabel
  suspicious_phrases: SuspiciousPhrase[]
  issues: string[]
  claims: Array<{
    claim: string
    verdict: ClaimVerdict
    explanation: string
  }>
}

export interface ArticleSummary {
  overall_risk_score: number
  credibility_score: number
  label: RiskLabel
  verdict_title: string
  conclusion: string
  style_risk_score: number
  style_label: RiskLabel
  fraud_risk_score: number
  factuality_risk_score: number | null
  factuality_label: FactualityLabel
  stats: {
    total_paragraphs: number
    dangerous_paragraphs: number
    suspicious_paragraphs: number
    safe_paragraphs: number
    total_claims: number
    contradicted_claims: number
    supported_claims: number
    unverified_claims: number
  }
}

export interface ArticleAnalysisResponse {
  article_id: string
  title: string
  summary: ArticleSummary
  paragraphs: ParagraphAnalysis[]
  claims: ClaimItem[]
  fact_check_error?: string | null
  fact_check_report_id?: string | null
  recommendations: string[]
}

export interface MediaAnalysisResult {
  analysisId: string
  modality: 'Image' | 'Audio'
  label: string
  riskScore: number
  confidence: number
  evidence: string[]
  detector: string
  input: string
  createdAt: string
  previewUrl?: string
  fileSize?: string
  details?: {
    anomalyType?: string
    deepfakeScore?: number
    metadataStatus?: string
    artifactsDetected?: string[]
    voiceCloneRisk?: number
    spectralAnomalies?: string[]
  }
}
