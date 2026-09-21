$ErrorActionPreference = 'Stop'
$baseUrl = 'http://127.0.0.1:8000'
$analysisId = 'sample-analysis-id'
try {
  Invoke-RestMethod -Method Get -Uri "$baseUrl/api/analyses/$analysisId" | ConvertTo-Json -Depth 20
} catch {
  Write-Host "GET /api/analyses/$analysisId returned a 404 or no record yet; this is expected before creating an analysis."
}
