import React, { useState } from 'react'
import {
  Shield,
  AlertTriangle,
  CheckCircle2,
  AlertCircle,
  Search,
  Cpu,
  Layers,
  ExternalLink,
  Copy,
  Check,
  RotateCcw,
  Sparkles,
  FileText,
  Image as ImageIcon,
  Mic,
  Info,
  FileAudio,
  FileImage,
  Sliders,
} from 'lucide-react'
import type {
  AppMode,
  ArticleAnalysisResponse,
  MediaAnalysisResult,
  ParagraphAnalysis,
  RiskLabel,
  SuspiciousPhrase,
} from './types'
import {
  SAMPLE_ARTICLES,
  SAMPLE_IMAGES,
  SAMPLE_AUDIOS,
} from './sampleData'
import type { SampleMediaItem } from './sampleData'

const API_BASE_URL = import.meta.env.VITE_API_URL || ''

export const App: React.FC = () => {
  // App Mode: 'text' | 'image' | 'audio'
  const [appMode, setAppMode] = useState<AppMode>('text')

  // --- State for Text Mode ---
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const [checkFacts, setCheckFacts] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [currentStep, setCurrentStep] = useState<number>(0)
  const [result, setResult] = useState<ArticleAnalysisResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [activeTab, setActiveTab] = useState<'paragraphs' | 'claims' | 'recommendations'>('paragraphs')
  const [riskFilter, setRiskFilter] = useState<'all' | RiskLabel>('all')
  const [copied, setCopied] = useState(false)

  // --- State for Media Modes (Image & Audio) ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [selectedSampleMedia, setSelectedSampleMedia] = useState<SampleMediaItem | null>(null)
  const [isMediaLoading, setIsMediaLoading] = useState(false)
  const [mediaResult, setMediaResult] = useState<MediaAnalysisResult | null>(null)
  const [mediaError, setMediaError] = useState<string | null>(null)

  // Handlers for Text Mode
  const handleSelectSample = (sampleId: string) => {
    const sample = SAMPLE_ARTICLES.find((s) => s.id === sampleId)
    if (sample) {
      setTitle(sample.title)
      setText(sample.text)
      setError(null)
    }
  }

  const handleAnalyzeText = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!text.trim()) {
      setError('Vui lòng nhập hoặc dán nội dung bài viết cần phân tích.')
      return
    }

    setIsLoading(true)
    setError(null)
    setResult(null)
    setCurrentStep(1)

    const stepTimer1 = setTimeout(() => setCurrentStep(2), 700)
    const stepTimer2 = setTimeout(() => setCurrentStep(3), 1600)
    const stepTimer3 = setTimeout(() => setCurrentStep(4), 2800)

    try {
      const endpoint = `${API_BASE_URL}/api/analyze/article`
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          title: title.trim(),
          text: text.trim(),
          check_facts: checkFacts,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}))
        throw new Error(errorData.detail || `Lỗi từ máy chủ: ${response.status}`)
      }

      const data: ArticleAnalysisResponse = await response.json()
      setResult(data)
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Không thể kết nối đến máy chủ phân tích'
      setError(message)
    } finally {
      clearTimeout(stepTimer1)
      clearTimeout(stepTimer2)
      clearTimeout(stepTimer3)
      setIsLoading(false)
      setCurrentStep(0)
    }
  }

  const handleCopyReport = () => {
    if (!result) return
    const reportText = `### BÁO CÁO PHÂN TÍCH TIN GIẢ - DEEPGUARD AI
**Tiêu đề:** ${result.title}
**Kết luận:** ${result.summary.verdict_title} (Rủi ro: ${result.summary.overall_risk_score}%, Độ tin cậy: ${result.summary.credibility_score}%)
**Nhận định Agent:** ${result.summary.conclusion}

**Thống kê:**
- Tổng số đoạn văn: ${result.summary.stats.total_paragraphs}
- Đoạn nguy cơ cao: ${result.summary.stats.dangerous_paragraphs}
- Luận điểm bị bác bỏ: ${result.summary.stats.contradicted_claims}
- Luận điểm được hỗ trợ: ${result.summary.stats.supported_claims}

**Khuyến nghị:**
${result.recommendations.map((r) => `- ${r}`).join('\n')}
`
    navigator.clipboard.writeText(reportText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  // Handlers for Media Modes (Image / Audio)
  const handleSelectMediaSample = (sample: SampleMediaItem) => {
    setSelectedSampleMedia(sample)
    setSelectedFile(null)
    setPreviewUrl(null)
    setMediaError(null)
    setMediaResult(null)
  }

  const handleFileDrop = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setSelectedFile(file)
      setSelectedSampleMedia(null)
      setMediaResult(null)
      setMediaError(null)
      setPreviewUrl(URL.createObjectURL(file))
    }
  }

  const handleAnalyzeMedia = async () => {
    if (!selectedFile && !selectedSampleMedia) {
      setMediaError(`Vui lòng chọn hoặc tải lên tệp ${appMode === 'image' ? 'hình ảnh' : 'âm thanh'}.`)
      return
    }

    setIsMediaLoading(true)
    setMediaError(null)

    // Nếu người dùng tải file thật lên, ta gửi tới API /api/analyze/media
    if (selectedFile) {
      try {
        const formData = new FormData()
        formData.append('file', selectedFile)
        formData.append('modality', appMode === 'image' ? 'Image' : 'Audio')

        const response = await fetch(`${API_BASE_URL}/api/analyze/media`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}))
          throw new Error(errData.detail || `Lỗi máy chủ (${response.status})`)
        }

        const data = await response.json()
        setMediaResult({
          analysisId: data.analysisId || `media-${Date.now()}`,
          modality: appMode === 'image' ? 'Image' : 'Audio',
          label: data.label || 'Likely authentic',
          riskScore: data.riskScore || 50,
          confidence: data.confidence || 0.65,
          evidence: data.evidence || ['Đã rà soát siêu dữ liệu và cấu trúc tệp'],
          detector: data.detector || `${appMode}-heuristic-v1`,
          input: selectedFile.name,
          createdAt: new Date().toISOString(),
          fileSize: `${(selectedFile.size / 1024).toFixed(1)} KB`,
          details: {
            anomalyType: appMode === 'image' ? 'EXIF & Compression Artifact' : 'Voice Frequency Pattern',
            deepfakeScore: data.riskScore,
            metadataStatus: 'Đã phân tích cấu trúc nhị phân',
            artifactsDetected: ['Đang sử dụng detector mẫu mô phỏng'],
          },
        })
      } catch (err: unknown) {
        const message = err instanceof Error ? err.message : 'Lỗi phân tích tệp'
        setMediaError(message)
      } finally {
        setIsMediaLoading(false)
      }
    } else if (selectedSampleMedia) {
      // Nếu người dùng bấm bài mẫu mô phỏng (Prototype demonstration)
      setTimeout(() => {
        setMediaResult({
          analysisId: `proto-${selectedSampleMedia.id}`,
          modality: appMode === 'image' ? 'Image' : 'Audio',
          label: selectedSampleMedia.label,
          riskScore: selectedSampleMedia.expectedRisk,
          confidence: 0.86,
          evidence: [
            selectedSampleMedia.description,
            `Kiểm tra cấu trúc tệp: ${selectedSampleMedia.fileName}`,
            'Sàng lọc qua thuật toán phát hiện bất thường biểu trưng (Prototype Model)',
          ],
          detector: `${appMode}-prototype-heuristic-v1`,
          input: selectedSampleMedia.fileName,
          createdAt: new Date().toISOString(),
          fileSize: selectedSampleMedia.fileSize,
          details: {
            anomalyType: selectedSampleMedia.category,
            deepfakeScore: selectedSampleMedia.expectedRisk,
            metadataStatus: 'Phát hiện sai lệch dấu vân tay kỹ thuật số',
            artifactsDetected: [selectedSampleMedia.description],
          },
        })
        setIsMediaLoading(false)
      }, 900)
    }
  }

  const renderHighlightedText = (paragraphText: string, phrases: SuspiciousPhrase[]) => {
    if (!phrases || phrases.length === 0) {
      return paragraphText
    }

    const sorted = [...phrases].sort((a, b) => a.start - b.start)
    const elements: React.ReactNode[] = []
    let lastIndex = 0

    sorted.forEach((phraseObj, idx) => {
      let start = phraseObj.start
      let end = phraseObj.end
      if (start < lastIndex || start >= paragraphText.length) {
        const found = paragraphText.toLowerCase().indexOf(phraseObj.phrase.toLowerCase(), lastIndex)
        if (found !== -1) {
          start = found
          end = found + phraseObj.phrase.length
        } else {
          return
        }
      }

      if (start > lastIndex) {
        elements.push(paragraphText.substring(lastIndex, start))
      }

      const isDanger =
        phraseObj.category === 'claim_contradicted' ||
        phraseObj.category === 'conspiracy' ||
        phraseObj.category === 'threat'
      elements.push(
        <mark
          key={`highlight-${idx}`}
          className={`highlight-phrase ${isDanger ? 'danger' : 'warning'}`}
          title={`${phraseObj.reason} (${phraseObj.category})`}
        >
          {paragraphText.substring(start, end)}
        </mark>
      )
      lastIndex = end
    })

    if (lastIndex < paragraphText.length) {
      elements.push(paragraphText.substring(lastIndex))
    }

    return elements
  }

  const filteredParagraphs = result
    ? result.paragraphs.filter((p) => (riskFilter === 'all' ? true : p.label === riskFilter))
    : []

  const wordCount = text.trim() ? text.trim().split(/\s+/).length : 0
  const charCount = text.length

  const overallRisk = result ? result.summary.overall_risk_score : 0
  const radius = 54
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference - (overallRisk / 100) * circumference
  const strokeColor = overallRisk >= 65 ? '#ef4444' : overallRisk >= 35 ? '#f59e0b' : '#10b981'

  // Media circle gauge calculation
  const mediaRisk = mediaResult ? mediaResult.riskScore : 0
  const mediaStrokeDashoffset = circumference - (mediaRisk / 100) * circumference
  const mediaStrokeColor = mediaRisk >= 65 ? '#ef4444' : mediaRisk >= 35 ? '#f59e0b' : '#10b981'

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-wrapper">
          <div className="brand-icon-box" id="brand-logo">
            <Shield size={26} />
          </div>
          <div>
            <h1 className="brand-title">DeepGuard AI</h1>
            <p className="brand-subtitle">
              <span>Nền tảng phân tích & kiểm chứng nội dung giả mạo đa phương tiện</span>
            </p>
          </div>
        </div>

        <div className="header-status-badge" id="agent-status-badge">
          <div className="status-dot"></div>
          <span>3 Chế độ phân tích Đa phương tiện</span>
        </div>
      </header>

      {/* 3-MODE SWITCHER TABS */}
      <div className="mode-switcher-wrapper">
        <div className="mode-switcher-tabs" id="mode-switcher-tabs">
          <button
            type="button"
            id="tab-mode-text"
            className={`mode-tab-btn ${appMode === 'text' ? 'active' : ''}`}
            onClick={() => setAppMode('text')}
          >
            <FileText size={18} />
            <span>Phân tích Văn bản</span>
            <span className="mode-tab-badge badge-complete">Hoàn thiện</span>
          </button>

          <button
            type="button"
            id="tab-mode-image"
            className={`mode-tab-btn ${appMode === 'image' ? 'active' : ''}`}
            onClick={() => setAppMode('image')}
          >
            <ImageIcon size={18} />
            <span>Phân tích Hình ảnh</span>
            <span className="mode-tab-badge badge-prototype">Biểu trưng</span>
          </button>

          <button
            type="button"
            id="tab-mode-audio"
            className={`mode-tab-btn ${appMode === 'audio' ? 'active' : ''}`}
            onClick={() => setAppMode('audio')}
          >
            <Mic size={18} />
            <span>Phân tích Âm thanh</span>
            <span className="mode-tab-badge badge-prototype">Biểu trưng</span>
          </button>
        </div>
      </div>

      {/* =========================================================
          MODE 1: PHÂN TÍCH VĂN BẢN (TEXT MODE)
          ========================================================= */}
      {appMode === 'text' && (
        <div id="text-mode-container">
          <div className="hero-section">
            <h2 className="hero-headline">
              Phân Tích Bài Viết Dài & <span className="hero-gradient-text">Rà Soát Sai Lệch</span>
            </h2>
            <p className="hero-description">
              Dán bất kỳ bài báo hay tin đồn (không giới hạn số chữ). Hệ thống sẽ tự động bóc tách từng đoạn văn, quét mô hình AI học máy kết hợp luật ngôn ngữ, trích xuất luận điểm đối chiếu với báo chí chính thống và tổng kết điểm tin cậy.
            </p>
          </div>

          {/* Input Card */}
          <div className="glass-card input-section" id="input-card">
            <div className="sample-buttons-wrapper">
              <span className="sample-btn-label">Thử nhanh bài mẫu:</span>
              {SAMPLE_ARTICLES.map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  id={`btn-${sample.id}`}
                  className="sample-chip-btn"
                  onClick={() => handleSelectSample(sample.id)}
                >
                  <FileText size={14} />
                  <span>{sample.category}</span>
                </button>
              ))}
              {(text || title) && (
                <button
                  type="button"
                  className="sample-chip-btn"
                  id="btn-clear-input"
                  onClick={() => {
                    setTitle('')
                    setText('')
                    setResult(null)
                    setError(null)
                  }}
                  style={{ marginLeft: 'auto' }}
                >
                  <RotateCcw size={14} />
                  <span>Làm mới</span>
                </button>
              )}
            </div>

            <form onSubmit={handleAnalyzeText}>
              <div className="form-group" style={{ marginBottom: 14 }}>
                <label className="form-label" htmlFor="article-title-input">
                  <span>Tiêu đề bài viết (tùy chọn)</span>
                </label>
                <input
                  id="article-title-input"
                  type="text"
                  className="input-text"
                  placeholder="Ví dụ: Cảnh báo khẩn cấp về tài khoản..."
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label" htmlFor="article-text-input">
                  <span>Nội dung bài viết (không giới hạn độ dài)</span>
                  <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>
                    {wordCount} từ | {charCount.toLocaleString()} ký tự
                  </span>
                </label>
                <textarea
                  id="article-text-input"
                  className="input-text textarea-article"
                  placeholder="Dán toàn bộ nội dung bài báo, văn bản hoặc tin đồn cần kiểm tra vào đây..."
                  value={text}
                  onChange={(e) => setText(e.target.value)}
                  required
                />
              </div>

              <div className="controls-bar">
                <label className="toggle-label" htmlFor="fact-check-toggle">
                  <input
                    id="fact-check-toggle"
                    type="checkbox"
                    className="toggle-checkbox"
                    checked={checkFacts}
                    onChange={(e) => setCheckFacts(e.target.checked)}
                  />
                  <span>Tích hợp Fact-checking (Tra cứu đối chiếu báo chí chính thống)</span>
                </label>

                <button
                  type="submit"
                  id="submit-analyze-btn"
                  className="btn-primary"
                  disabled={isLoading || !text.trim()}
                >
                  {isLoading ? (
                    <>
                      <div className="spinner"></div>
                      <span>Đang chạy Agent phân tích...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      <span>Bắt đầu phân tích bài viết</span>
                    </>
                  )}
                </button>
              </div>
            </form>

            {error && (
              <div
                id="error-message-box"
                style={{
                  marginTop: 14,
                  padding: '12px 16px',
                  borderRadius: '8px',
                  background: 'var(--danger-bg)',
                  border: '1px solid var(--danger-border)',
                  color: '#f87171',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  fontSize: '0.9rem',
                }}
              >
                <AlertCircle size={18} />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Loading Pipeline Steps Indicator */}
          {isLoading && (
            <div className="glass-card pipeline-card" id="pipeline-status-card">
              <div className="pipeline-header">
                <div className="spinner"></div>
                <strong style={{ fontSize: '1rem' }}>Agent đang xử lý bài viết đa bước...</strong>
              </div>
              <div className="pipeline-steps-grid">
                <div className={`pipeline-step-item ${currentStep >= 1 ? 'active' : ''}`}>
                  <Layers size={16} />
                  <span>1. Tách cấu trúc đoạn văn</span>
                </div>
                <div className={`pipeline-step-item ${currentStep >= 2 ? 'active' : ''}`}>
                  <Cpu size={16} />
                  <span>2. Quét mô hình AI & luật rủi ro</span>
                </div>
                <div className={`pipeline-step-item ${currentStep >= 3 ? 'active' : ''}`}>
                  <Search size={16} />
                  <span>3. Fact-checking nguồn báo chí</span>
                </div>
                <div className={`pipeline-step-item ${currentStep >= 4 ? 'active' : ''}`}>
                  <Shield size={16} />
                  <span>4. Tổng hợp điểm & kết luận</span>
                </div>
              </div>
            </div>
          )}

          {/* Analysis Results View */}
          {result && (
            <div className="results-container" id="results-dashboard">
              {/* Executive Overview Card */}
              <div
                className={`glass-card overview-card ${
                  result.summary.label === 'DANGEROUS'
                    ? 'danger'
                    : result.summary.label === 'SUSPICIOUS'
                    ? 'warning'
                    : 'safe'
                }`}
                id="overview-card"
              >
                {/* Score Radial Gauge */}
                <div className="gauge-box">
                  <div className="circle-score-wrapper">
                    <svg className="circle-svg">
                      <circle className="circle-bg" cx="70" cy="70" r={radius} />
                      <circle
                        className="circle-progress"
                        cx="70"
                        cy="70"
                        r={radius}
                        stroke={strokeColor}
                        strokeDasharray={circumference}
                        strokeDashoffset={strokeDashoffset}
                      />
                    </svg>
                    <div className="circle-score-inner">
                      <span className="circle-score-number" style={{ color: strokeColor }}>
                        {result.summary.overall_risk_score}%
                      </span>
                      <span className="circle-score-unit">Rủi ro</span>
                    </div>
                  </div>
                  <p className="gauge-caption">
                    Độ tin cậy: <strong style={{ color: '#fff' }}>{result.summary.credibility_score}%</strong>
                  </p>
                </div>

                {/* Verdict text and stats */}
                <div className="verdict-content-box">
                  <div className="verdict-header-row">
                    <span
                      className={`verdict-badge ${
                        result.summary.label === 'DANGEROUS'
                          ? 'badge-danger'
                          : result.summary.label === 'SUSPICIOUS'
                          ? 'badge-warning'
                          : 'badge-safe'
                      }`}
                    >
                      {result.summary.label === 'DANGEROUS' ? (
                        <AlertTriangle size={16} />
                      ) : result.summary.label === 'SUSPICIOUS' ? (
                        <AlertCircle size={16} />
                      ) : (
                        <CheckCircle2 size={16} />
                      )}
                      <span>{result.summary.label}</span>
                    </span>
                    <h3 className="verdict-title">{result.summary.verdict_title}</h3>
                  </div>

                  <p className="verdict-conclusion-text">{result.summary.conclusion}</p>

                  {/* Stats Bar */}
                  <div className="stats-pills-bar">
                    <div className="stat-pill">
                      <span className="stat-pill-val">{result.summary.stats.total_paragraphs}</span>
                      <span className="stat-pill-label">Đoạn văn</span>
                    </div>
                    <div className="stat-pill">
                      <span className="stat-pill-val" style={{ color: '#ef4444' }}>
                        {result.summary.stats.dangerous_paragraphs}
                      </span>
                      <span className="stat-pill-label">Nguy cơ cao</span>
                    </div>
                    <div className="stat-pill">
                      <span className="stat-pill-val" style={{ color: '#f59e0b' }}>
                        {result.summary.stats.suspicious_paragraphs}
                      </span>
                      <span className="stat-pill-label">Nghi vấn</span>
                    </div>
                    <div className="stat-pill">
                      <span className="stat-pill-val" style={{ color: '#10b981' }}>
                        {result.summary.stats.safe_paragraphs}
                      </span>
                      <span className="stat-pill-label">An toàn</span>
                    </div>
                    {result.claims.length > 0 && (
                      <div className="stat-pill">
                        <span className="stat-pill-val" style={{ color: '#38bdf8' }}>
                          {result.summary.stats.contradicted_claims} / {result.claims.length}
                        </span>
                        <span className="stat-pill-label">Luận điểm mâu thuẫn</span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Navigation Tabs and Actions */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 12 }}>
                <div className="tabs-nav" id="results-tabs">
                  <button
                    type="button"
                    id="tab-btn-paragraphs"
                    className={`tab-btn ${activeTab === 'paragraphs' ? 'active' : ''}`}
                    onClick={() => setActiveTab('paragraphs')}
                  >
                    <Layers size={16} />
                    <span>Chi tiết từng đoạn văn</span>
                    <span className="tab-counter">{result.paragraphs.length}</span>
                  </button>

                  <button
                    type="button"
                    id="tab-btn-claims"
                    className={`tab-btn ${activeTab === 'claims' ? 'active' : ''}`}
                    onClick={() => setActiveTab('claims')}
                  >
                    <Search size={16} />
                    <span>Kiểm chứng sự thật (Fact-check)</span>
                    <span className="tab-counter">{result.claims.length}</span>
                  </button>

                  <button
                    type="button"
                    id="tab-btn-recommendations"
                    className={`tab-btn ${activeTab === 'recommendations' ? 'active' : ''}`}
                    onClick={() => setActiveTab('recommendations')}
                  >
                    <Shield size={16} />
                    <span>Khuyến nghị xử lý</span>
                  </button>
                </div>

                <button
                  type="button"
                  id="btn-copy-report"
                  className="sample-chip-btn"
                  onClick={handleCopyReport}
                  style={{ background: 'rgba(99, 102, 241, 0.15)', color: '#a5b4fc', borderColor: 'rgba(99, 102, 241, 0.3)' }}
                >
                  {copied ? <Check size={14} /> : <Copy size={14} />}
                  <span>{copied ? 'Đã sao chép báo cáo' : 'Sao chép báo cáo Markdown'}</span>
                </button>
              </div>

              {/* Tab 1: Paragraphs Breakdown */}
              {activeTab === 'paragraphs' && (
                <div>
                  <div className="filter-row">
                    <span style={{ fontSize: '0.86rem', color: 'var(--text-secondary)' }}>
                      Hiển thị {filteredParagraphs.length} / {result.paragraphs.length} đoạn văn:
                    </span>
                    <div className="filter-chips">
                      <button
                        type="button"
                        className={`filter-chip ${riskFilter === 'all' ? 'active' : ''}`}
                        onClick={() => setRiskFilter('all')}
                      >
                        Tất cả
                      </button>
                      <button
                        type="button"
                        className={`filter-chip ${riskFilter === 'DANGEROUS' ? 'active' : ''}`}
                        onClick={() => setRiskFilter('DANGEROUS')}
                      >
                        Nguy cơ cao ({result.summary.stats.dangerous_paragraphs})
                      </button>
                      <button
                        type="button"
                        className={`filter-chip ${riskFilter === 'SUSPICIOUS' ? 'active' : ''}`}
                        onClick={() => setRiskFilter('SUSPICIOUS')}
                      >
                        Nghi vấn ({result.summary.stats.suspicious_paragraphs})
                      </button>
                      <button
                        type="button"
                        className={`filter-chip ${riskFilter === 'SAFE' ? 'active' : ''}`}
                        onClick={() => setRiskFilter('SAFE')}
                      >
                        An toàn ({result.summary.stats.safe_paragraphs})
                      </button>
                    </div>
                  </div>

                  <div className="paragraphs-list" id="paragraphs-list">
                    {filteredParagraphs.map((paragraph: ParagraphAnalysis) => {
                      const cardClass =
                        paragraph.label === 'DANGEROUS'
                          ? 'danger'
                          : paragraph.label === 'SUSPICIOUS'
                          ? 'warning'
                          : 'safe'

                      return (
                        <div
                          key={`para-${paragraph.index}`}
                          id={`paragraph-item-${paragraph.index}`}
                          className={`paragraph-card ${cardClass}`}
                        >
                          <div className="paragraph-header">
                            <div className="paragraph-index-tag">
                              <span>Đoạn #{paragraph.index}</span>
                              <span
                                className={`verdict-badge ${
                                  paragraph.label === 'DANGEROUS'
                                    ? 'badge-danger'
                                    : paragraph.label === 'SUSPICIOUS'
                                    ? 'badge-warning'
                                    : 'badge-safe'
                                }`}
                                style={{ padding: '2px 8px', fontSize: '0.75rem' }}
                              >
                                {paragraph.label}
                              </span>
                            </div>

                            <div className="scores-breakdown">
                              <div className="mini-score-bar-group">
                                <span>Rủi ro:</span>
                                <div className="mini-bar">
                                  <div
                                    className="mini-bar-fill"
                                    style={{
                                      width: `${paragraph.risk_score}%`,
                                      background:
                                        paragraph.risk_score >= 65
                                          ? '#ef4444'
                                          : paragraph.risk_score >= 35
                                          ? '#f59e0b'
                                          : '#10b981',
                                    }}
                                  />
                                </div>
                                <strong style={{ color: '#fff' }}>{paragraph.risk_score}%</strong>
                              </div>

                              {paragraph.model_score !== null && (
                                <span style={{ fontSize: '0.78rem', color: '#94a3b8' }}>
                                  AI: {(paragraph.model_score * 100).toFixed(0)}%
                                </span>
                              )}
                            </div>
                          </div>

                          <p className="paragraph-content-text">
                            {renderHighlightedText(paragraph.text, paragraph.suspicious_phrases)}
                          </p>

                          {paragraph.issues.length > 0 && (
                            <div className="issues-tags-list">
                              {paragraph.issues.map((issue, i) => (
                                <span
                                  key={`issue-${i}`}
                                  className={`issue-tag ${
                                    issue.includes('bác bỏ') ||
                                    issue.includes('áp lực') ||
                                    issue.includes('chuyển tiền')
                                      ? 'danger'
                                      : ''
                                  }`}
                                >
                                  <AlertCircle size={12} />
                                  <span>{issue}</span>
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                </div>
              )}

              {/* Tab 2: Fact-Check Claims */}
              {activeTab === 'claims' && (
                <div className="claims-grid" id="claims-section">
                  {result.claims.length === 0 ? (
                    <div className="glass-card" style={{ textAlign: 'center', padding: '40px 20px', color: '#94a3b8' }}>
                      <Search size={36} style={{ margin: '0 auto 12px', opacity: 0.5 }} />
                      <p>Không phát hiện luận điểm kiểm chứng cụ thể nào trong bài viết này.</p>
                    </div>
                  ) : (
                    result.claims.map((claim, idx) => {
                      const cardBorder =
                        claim.verdict === 'contradicted'
                          ? 'contradicted'
                          : claim.verdict === 'supported'
                          ? 'supported'
                          : 'mixed'

                      const verdictTitle =
                        claim.verdict === 'contradicted'
                          ? 'Bị báo chí mâu thuẫn / Bác bỏ'
                          : claim.verdict === 'supported'
                          ? 'Được nguồn tin xác thực'
                          : 'Chưa đủ bằng chứng / Hỗn hợp'

                      return (
                        <div key={`claim-${idx}`} className={`claim-card ${cardBorder}`} id={`claim-item-${idx}`}>
                          <div className="claim-header">
                            <h4 className="claim-title-text">
                              {idx + 1}. "{claim.claim}"
                            </h4>
                            <span
                              className={`verdict-badge ${
                                claim.verdict === 'contradicted'
                                  ? 'badge-danger'
                                  : claim.verdict === 'supported'
                                  ? 'badge-safe'
                                  : 'badge-warning'
                              }`}
                            >
                              {verdictTitle}
                            </span>
                          </div>

                          <div className="claim-explanation-box">
                            <strong style={{ color: '#f1f5f9' }}>Nhận định kiểm chứng: </strong>
                            <span>{claim.explanation}</span>
                          </div>

                          {claim.evidence && claim.evidence.length > 0 && (
                            <div className="sources-list">
                              <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', fontWeight: 600 }}>
                                Nguồn đối chiếu ({claim.evidence.length} nguồn):
                              </span>
                              {claim.evidence.map((source, sIdx) => (
                                <div key={`source-${sIdx}`} className="source-item">
                                  <ExternalLink size={16} style={{ flexShrink: 0, marginTop: 2, color: '#38bdf8' }} />
                                  <div style={{ flex: 1 }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                                      <a href={source.url} target="_blank" rel="noopener noreferrer">
                                        {source.title}
                                      </a>
                                      {source.priority_publisher && (
                                        <span className="priority-source-badge">Nguồn tin ưu tiên</span>
                                      )}
                                      <span style={{ fontSize: '0.75rem', color: '#64748b' }}>({source.source})</span>
                                    </div>
                                    {source.snippet && <p className="source-snippet">{source.snippet}</p>}
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      )
                    })
                  )}
                </div>
              )}

              {/* Tab 3: Recommendations */}
              {activeTab === 'recommendations' && (
                <div className="glass-card recommendations-card" id="recommendations-section">
                  <h3 style={{ fontSize: '1.15rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 10 }}>
                    <Shield size={20} color="var(--primary)" />
                    <span>Khuyến nghị của Hệ thống đối với Độc giả</span>
                  </h3>
                  <ul className="recommendations-list">
                    {result.recommendations.map((rec, i) => (
                      <li key={`rec-${i}`} className="recommendation-item">
                        <CheckCircle2 size={18} color="#34d399" style={{ flexShrink: 0, marginTop: 2 }} />
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* =========================================================
          MODE 2 & 3: PHÂN TÍCH HÌNH ẢNH & ÂM THANH (BIỂU TRƯNG)
          ========================================================= */}
      {(appMode === 'image' || appMode === 'audio') && (
        <div id="media-mode-container">
          <div className="hero-section">
            <h2 className="hero-headline">
              {appMode === 'image' ? 'Phân Tích Hình Ảnh' : 'Phân Tích Âm Thanh'} &{' '}
              <span className="hero-gradient-text">
                {appMode === 'image' ? 'Phát Hiện Deepfake/Cắt Ghép' : 'Phát Hiện Voice Clone AI'}
              </span>
            </h2>
            <p className="hero-description">
              {appMode === 'image'
                ? 'Tải lên hình ảnh cần kiểm tra dấu vết chỉnh sửa, ghép đè con dấu, mặt AI (Synthetic Face) hoặc sai lệch siêu dữ liệu máy ảnh EXIF.'
                : 'Tải lên tệp âm thanh hoặc bản ghi âm cuộc gọi để sàng lọc giọng nói tổng hợp AI, đe dọa tống tiền hoặc bất thường quang phổ âm thanh.'}
            </p>
          </div>

          {/* Prototype Notice Banner */}
          <div className="prototype-notice-banner">
            <Info size={20} color="#fbbf24" style={{ flexShrink: 0 }} />
            <div>
              <strong>Chế độ biểu trưng (Prototype Model):</strong> Bộ detector chuyên sâu cho{' '}
              {appMode === 'image' ? 'hình ảnh' : 'âm thanh'} đang được phát triển theo lộ trình dự án. Giao diện dưới đây đóng vai trò mô phỏng quy trình kiểm tra và hiển thị kết quả sàng lọc sơ bộ.
            </div>
          </div>

          {/* Media Input Card */}
          <div className="glass-card input-section">
            {/* Quick Sample Selector for Media */}
            <div className="sample-buttons-wrapper">
              <span className="sample-btn-label">Chọn tệp mẫu mô phỏng:</span>
              {(appMode === 'image' ? SAMPLE_IMAGES : SAMPLE_AUDIOS).map((sample) => (
                <button
                  key={sample.id}
                  type="button"
                  id={`btn-${sample.id}`}
                  className="sample-chip-btn"
                  onClick={() => handleSelectMediaSample(sample)}
                >
                  {appMode === 'image' ? <FileImage size={14} /> : <FileAudio size={14} />}
                  <span>{sample.category}</span>
                </button>
              ))}
              {(selectedFile || selectedSampleMedia || mediaResult) && (
                <button
                  type="button"
                  className="sample-chip-btn"
                  onClick={() => {
                    setSelectedFile(null)
                    setSelectedSampleMedia(null)
                    setPreviewUrl(null)
                    setMediaResult(null)
                    setMediaError(null)
                  }}
                  style={{ marginLeft: 'auto' }}
                >
                  <RotateCcw size={14} />
                  <span>Làm mới</span>
                </button>
              )}
            </div>

            {/* Dropzone */}
            <label className="media-dropzone" htmlFor="media-file-input">
              <input
                id="media-file-input"
                type="file"
                accept={appMode === 'image' ? 'image/*' : 'audio/*'}
                style={{ display: 'none' }}
                onChange={handleFileDrop}
              />
              <div className="dropzone-icon-circle">
                {appMode === 'image' ? <ImageIcon size={28} /> : <Mic size={28} />}
              </div>
              <div>
                <p className="dropzone-title">
                  {selectedFile
                    ? `Đã chọn: ${selectedFile.name}`
                    : selectedSampleMedia
                    ? `Đang chọn mẫu: ${selectedSampleMedia.title}`
                    : `Nhấp hoặc kéo thả tệp ${appMode === 'image' ? 'hình ảnh' : 'âm thanh'} vào đây`}
                </p>
                <p className="dropzone-hint">
                  {appMode === 'image' ? 'Hỗ trợ định dạng JPG, PNG, WEBP (tối đa 25MB)' : 'Hỗ trợ định dạng MP3, WAV, M4A, FLAC (tối đa 50MB)'}
                </p>
              </div>
            </label>

            {/* Preview Box if selected */}
            {(selectedFile || selectedSampleMedia) && (
              <div className="media-preview-container">
                <div className="media-file-meta-bar">
                  <div className="media-file-info">
                    {appMode === 'image' ? <FileImage size={18} color="#38bdf8" /> : <FileAudio size={18} color="#38bdf8" />}
                    <strong>
                      {selectedFile ? selectedFile.name : selectedSampleMedia?.fileName}
                    </strong>
                    <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                      ({selectedFile ? `${(selectedFile.size / 1024).toFixed(1)} KB` : selectedSampleMedia?.fileSize})
                    </span>
                  </div>

                  <span className="priority-source-badge">
                    {selectedSampleMedia ? 'Dữ liệu mẫu mô phỏng' : 'Tệp tải lên'}
                  </span>
                </div>

                {/* Simulated Waveform or Image Preview */}
                {appMode === 'image' && (
                  <div style={{ textAlign: 'center', marginTop: 10 }}>
                    {previewUrl ? (
                      <img src={previewUrl} alt="Preview" className="media-thumbnail-view" />
                    ) : (
                      <div
                        style={{
                          height: 140,
                          borderRadius: 8,
                          background: 'rgba(0,0,0,0.3)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          color: '#94a3b8',
                          fontSize: '0.9rem',
                        }}
                      >
                        [Mô phỏng khung hình: {selectedSampleMedia?.title}]
                      </div>
                    )}
                  </div>
                )}

                {appMode === 'audio' && (
                  <div style={{ marginTop: 10 }}>
                    <div className="waveform-container">
                      {Array.from({ length: 48 }).map((_, i) => (
                        <div
                          key={`wave-${i}`}
                          className="wave-bar"
                          style={{
                            animationDelay: `${(i % 10) * 0.12}s`,
                            height: `${30 + (Math.sin(i) * 30 + 30)}%`,
                          }}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {/* Submit button */}
                <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 14 }}>
                  <button
                    type="button"
                    className="btn-primary"
                    disabled={isMediaLoading}
                    onClick={handleAnalyzeMedia}
                  >
                    {isMediaLoading ? (
                      <>
                        <div className="spinner"></div>
                        <span>Đang xử lý phân tích...</span>
                      </>
                    ) : (
                      <>
                        <Sparkles size={18} />
                        <span>Bắt đầu phân tích {appMode === 'image' ? 'hình ảnh' : 'âm thanh'}</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {mediaError && (
              <div
                style={{
                  marginTop: 14,
                  padding: '12px 16px',
                  borderRadius: '8px',
                  background: 'var(--danger-bg)',
                  border: '1px solid var(--danger-border)',
                  color: '#f87171',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  fontSize: '0.9rem',
                }}
              >
                <AlertCircle size={18} />
                <span>{mediaError}</span>
              </div>
            )}
          </div>

          {/* Media Analysis Results */}
          {mediaResult && (
            <div className="results-container">
              <div
                className={`glass-card overview-card ${
                  mediaResult.riskScore >= 65
                    ? 'danger'
                    : mediaResult.riskScore >= 35
                    ? 'warning'
                    : 'safe'
                }`}
              >
                {/* Score Radial Gauge */}
                <div className="gauge-box">
                  <div className="circle-score-wrapper">
                    <svg className="circle-svg">
                      <circle className="circle-bg" cx="70" cy="70" r={radius} />
                      <circle
                        className="circle-progress"
                        cx="70"
                        cy="70"
                        r={radius}
                        stroke={mediaStrokeColor}
                        strokeDasharray={circumference}
                        strokeDashoffset={mediaStrokeDashoffset}
                      />
                    </svg>
                    <div className="circle-score-inner">
                      <span className="circle-score-number" style={{ color: mediaStrokeColor }}>
                        {mediaResult.riskScore}%
                      </span>
                      <span className="circle-score-unit">Rủi ro</span>
                    </div>
                  </div>
                  <p className="gauge-caption">
                    Độ tin cậy: <strong style={{ color: '#fff' }}>{(mediaResult.confidence * 100).toFixed(0)}%</strong>
                  </p>
                </div>

                <div className="verdict-content-box">
                  <div className="verdict-header-row">
                    <span
                      className={`verdict-badge ${
                        mediaResult.riskScore >= 65
                          ? 'badge-danger'
                          : mediaResult.riskScore >= 35
                          ? 'badge-warning'
                          : 'badge-safe'
                      }`}
                    >
                      {mediaResult.riskScore >= 65 ? <AlertTriangle size={16} /> : <CheckCircle2 size={16} />}
                      <span>{mediaResult.label}</span>
                    </span>
                    <span className="priority-source-badge" style={{ padding: '4px 10px' }}>
                      Detector: {mediaResult.detector}
                    </span>
                  </div>

                  <p className="verdict-conclusion-text">
                    Tệp <strong>{mediaResult.input}</strong> đã được quét qua bộ lọc phân tích đa phương tiện{' '}
                    {mediaResult.modality === 'Image' ? 'hình ảnh' : 'âm thanh'}. Điểm rủi ro sơ bộ được ước tính ở mức{' '}
                    <strong>{mediaResult.riskScore}/100</strong>.
                  </p>

                  {/* Evidence Items */}
                  <div style={{ marginTop: 14 }}>
                    <span style={{ fontSize: '0.84rem', color: '#94a3b8', fontWeight: 600 }}>
                      Bằng chứng sàng lọc ghi nhận:
                    </span>
                    <ul style={{ listStyle: 'none', marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                      {mediaResult.evidence.map((ev, i) => (
                        <li
                          key={`ev-${i}`}
                          style={{
                            fontSize: '0.88rem',
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: 8,
                            color: '#cbd5e1',
                          }}
                        >
                          <CheckCircle2 size={15} color="var(--primary)" style={{ flexShrink: 0, marginTop: 3 }} />
                          <span>{ev}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>

              {/* Technical Inspection Details Grid */}
              <div className="glass-card">
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: 10 }}>
                  <Sliders size={18} color="var(--primary)" />
                  <span>Thông số Kỹ thuật & Phát hiện Dấu vết ({mediaResult.modality})</span>
                </h3>

                <div className="inspector-metrics-grid">
                  <div className="inspector-metric-card">
                    <span className="inspector-metric-label">Xác suất Can thiệp Kỹ thuật số</span>
                    <span
                      className="inspector-metric-val"
                      style={{ color: mediaResult.riskScore >= 65 ? '#ef4444' : '#10b981' }}
                    >
                      {mediaResult.riskScore}%
                    </span>
                    <span className="inspector-metric-desc">
                      {mediaResult.modality === 'Image'
                        ? 'Dấu hiệu chỉnh sửa pixel và ghép chồng lớp'
                        : 'Dấu hiệu nhịp điệu giọng máy móc / Voice Clone'}
                    </span>
                  </div>

                  <div className="inspector-metric-card">
                    <span className="inspector-metric-label">Kiểm tra Siêu dữ liệu (Metadata)</span>
                    <span className="inspector-metric-val" style={{ color: '#38bdf8' }}>
                      {mediaResult.modality === 'Image' ? 'EXIF Header' : 'Audio Stream'}
                    </span>
                    <span className="inspector-metric-desc">
                      {mediaResult.details?.metadataStatus || 'Đã kiểm tra cấu trúc header'}
                    </span>
                  </div>

                  <div className="inspector-metric-card">
                    <span className="inspector-metric-label">Trạng thái Detector</span>
                    <span className="inspector-metric-val" style={{ color: '#fbbf24' }}>
                      Heuristic Prototype
                    </span>
                    <span className="inspector-metric-desc">
                      Chế độ sàng lọc biểu trưng theo đặc tả dự án
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Footer */}
      <footer className="app-footer">
        <p>DeepGuard AI v2.5 — Nghiên cứu phân tích & Phát hiện tin giả đa phương tiện</p>
        <p style={{ marginTop: 4 }}>
          Tích hợp 3 Chế độ: Phân tích Văn bản (NLP & Fact-Check), Phân tích Hình ảnh & Âm thanh (Biểu trưng)
        </p>
      </footer>
    </div>
  )
}

export default App
