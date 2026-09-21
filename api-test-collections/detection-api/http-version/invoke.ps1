$ErrorActionPreference = 'Stop'
$baseUrl = 'http://127.0.0.1:8000'
$resp = Invoke-RestMethod -Method Get -Uri "$baseUrl/api/version"
$resp | ConvertTo-Json -Depth 10
