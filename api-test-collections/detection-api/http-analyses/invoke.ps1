$ErrorActionPreference = 'Stop'
$baseUrl = 'http://127.0.0.1:8000'
Invoke-RestMethod -Method Get -Uri "$baseUrl/api/analyses" | ConvertTo-Json -Depth 20
