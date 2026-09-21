$ErrorActionPreference = 'Stop'
$baseUrl = 'http://127.0.0.1:8000'
$payload = @{ text = 'The committee approved the revised climate brief.' } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$baseUrl/api/analyze/text" -ContentType 'application/json' -Body $payload | ConvertTo-Json -Depth 20
