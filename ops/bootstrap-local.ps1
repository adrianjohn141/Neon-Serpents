param([string]$AdminPassword)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root '.env.docker'
if (Test-Path -LiteralPath $target) { throw '.env.docker already exists; refusing to overwrite it.' }
if ([string]::IsNullOrWhiteSpace($AdminPassword)) {
  $secure = Read-Host 'Local administrator password' -AsSecureString
  $AdminPassword = [System.Net.NetworkCredential]::new('', $secure).Password
}
if ($AdminPassword.Length -lt 12) { throw 'Administrator password must contain at least 12 characters.' }
function New-Secret([int]$bytes) {
  $buffer = [byte[]]::new($bytes)
  $generator = [System.Security.Cryptography.RandomNumberGenerator]::Create()
  try { $generator.GetBytes($buffer) } finally { $generator.Dispose() }
  [Convert]::ToBase64String($buffer)
}
$bootstrapMount = "${PSScriptRoot}:/bootstrap:ro"
$adminHash = $AdminPassword | docker run --rm -i -v $bootstrapMount python:3.12-slim sh /bootstrap/hash-password.sh
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($adminHash)) { throw 'Could not hash the administrator password.' }
$adminHashB64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($adminHash.Trim()))
$lines = @(
  "POSTGRES_PASSWORD=$(New-Secret 32)",
  'MINIO_ROOT_USER=neon-local',
  "MINIO_ROOT_PASSWORD=$(New-Secret 32)",
  "ADMIN_PASSWORD_HASH_B64=$adminHashB64",
  "SESSION_SECRET=$(New-Secret 48)",
  "MANIFEST_SIGNING_SEED=$(New-Secret 32)",
  'ACTOR_COUNT=12',
  'COOKIE_SECURE=false',
  'LOCAL_BACKUP_PATH=./artifacts/backups',
  "RESTIC_PASSWORD=$(New-Secret 32)",
  'RESTIC_REPOSITORY=/repository'
)
[IO.File]::WriteAllLines($target, $lines, [Text.UTF8Encoding]::new($false))
Write-Host 'Created .env.docker with local secrets. Keep this file private.'
