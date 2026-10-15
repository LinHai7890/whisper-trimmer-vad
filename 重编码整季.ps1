#Requires -Version 5.1
<#
.SYNOPSIS
    一键批量重编码剪辑：把整个剧集文件夹剪成只保留有人声的片段。

.DESCRIPTION
    逐集调用 whisper_trim.py 的重编码路径，自动挑选显卡编码器，导出可回溯的
    剪辑表，并汇总每一集的保留时长与渲染耗时。

    与直接跑 whisper_trim.py "文件夹" 的区别在于本脚本会自己逐集调度，
    因此能给出每一集的耗时与结果，失败时也能指出具体是哪一集。

.PARAMETER InputPath
    剧集文件夹或单个文件。文件夹会递归扫描。

.PARAMETER OutputDir
    输出文件夹。省略时成品写在源文件旁边，名字加 _trimmed 后缀。
    指定时保持原有的子目录结构。

.PARAMETER ListDir
    剪辑表输出目录，每个源文件会生成 .ffconcat 和 .csv。
    因为输出是压缩后的时间轴，CSV 里的原片时间码是找回某段戏的唯一办法，
    建议每次都导出。

.PARAMETER Encoder
    重编码方式。auto 会优先选显卡编码器（NVIDIA 为 h264_nvenc），
    选不到退回 libx264。copy 是无损快速模式，但会多保留静音。

.PARAMETER Cq
    硬件编码器（nvenc/qsv）的画质，数字越大画质越差、体积越小。

.PARAMETER Crf
    libx264/libx265 的画质，数字越大画质越差、体积越小。

.PARAMETER VadThreshold
    语音检测阈值 0~1。台词被剪掉就调低（0.4），音乐被误留就调高（0.6）。

.PARAMETER TargetSize
    目标成品体积，如 700M。工具按实际保留时长反算视频码率，并自动切换到
    两遍编码。实测偏差在 3% 以内，耗时约为单遍的两倍。

.PARAMETER Bitrate
    目标视频码率，如 1500k。配合 -RateControl vbr 使用。

.PARAMETER RateControl
    码率控制方式。cq 恒定质量（体积不可控），vbr 单遍目标码率，
    2pass 两遍编码（体积最准，耗时约 1.8 倍）。

.EXAMPLE
    .\重编码整季.ps1 "D:\剧集"

.EXAMPLE
    .\重编码整季.ps1 "D:\剧集" -OutputDir "D:\剪辑结果" -ListDir "D:\剪辑表" -VadThreshold 0.4

.EXAMPLE
    .\重编码整季.ps1 "D:\剧集" -TargetSize 700M -ListDir "D:\剪辑表"

.EXAMPLE
    .\重编码整季.ps1 "D:\剧集" -DryRun
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory, Position = 0)]
    [string]$InputPath,

    [string]$OutputDir,
    [string]$ListDir,

    [ValidateSet('auto', 'copy', 'libx264', 'libx265', 'nvenc', 'hevc_nvenc', 'qsv', 'vaapi')]
    [string]$Encoder = 'auto',
    [int]$Cq = 21,
    [int]$Crf = 18,
    [string]$Preset = 'medium',

    [ValidateSet('cq', 'vbr', '2pass')]
    [string]$RateControl = 'cq',
    [string]$Bitrate,
    [string]$TargetSize,

    [ValidateRange(0.01, 0.99)]
    [double]$VadThreshold = 0.5,
    [int]$VadPadMs = 200,
    [int]$VadMinSilenceMs = 350,
    [int]$VadMinSpeechMs = 250,

    [switch]$Overwrite,
    [switch]$DryRun
)

# 注意：本文件必须保存为带 BOM 的 UTF-8，否则 Windows PowerShell 5.1 会按
# 系统 ANSI 代码页解析脚本里的中文，导致语法错误。

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Python 走管道时用的是区域编码，强制它跟控制台一致，
# 否则中文路径和输出会变成乱码。
$env:PYTHONIOENCODING = [Console]::OutputEncoding.WebName

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$TrimScript = Join-Path $ScriptRoot 'whisper_trim.py'

$MediaExtensions = @(
    '.3g2', '.3gp', '.avi', '.flv', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.webm', '.wmv',
    '.aac', '.ac3', '.aif', '.aiff', '.alac', '.amr', '.ape', '.au', '.flac', '.m4a', '.mka',
    '.mp3', '.oga', '.ogg', '.opus', '.ra', '.wav', '.wma', '.wv'
)

function Write-Step { param([string]$Message) Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Message) Write-Host "    $Message" -ForegroundColor Green }
function Write-Warn { param([string]$Message) Write-Host "    $Message" -ForegroundColor Yellow }
function Write-Bad  { param([string]$Message) Write-Host "    $Message" -ForegroundColor Red }

# ---------------------------------------------------------------- 环境检查

function Resolve-Python {
    $candidates = @(
        (Join-Path $ScriptRoot '.venv\Scripts\python.exe'),
        (Join-Path $ScriptRoot 'venv\Scripts\python.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    $fromPath = Get-Command python -ErrorAction SilentlyContinue
    if ($fromPath) { return $fromPath.Source }

    Write-Bad '找不到 Python。请先按使用教程.md 第二节创建 .venv 并安装依赖。'
    exit 1
}

function Assert-Tool {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        Write-Bad "找不到 $Name。请先安装 FFmpeg 并确保它在 PATH 里。"
        exit 1
    }
}

if (-not (Test-Path -LiteralPath $TrimScript)) {
    Write-Bad "找不到 $TrimScript，请在仓库目录下运行本脚本。"
    exit 1
}

Assert-Tool -Name 'ffmpeg'
Assert-Tool -Name 'ffprobe'

if (-not (Test-Path -LiteralPath $InputPath)) {
    Write-Bad "输入路径不存在：$InputPath"
    exit 1
}

$python = Resolve-Python
$source = (Resolve-Path -LiteralPath $InputPath).Path

Write-Step '环境'
Write-Ok "Python      $python"
Write-Ok "FFmpeg      $((Get-Command ffmpeg).Source)"
Write-Ok "输入        $source"
if ($OutputDir) { Write-Ok "输出        $((New-Item -ItemType Directory -Force -Path $OutputDir).FullName)" }
if ($ListDir) { Write-Ok "剪辑表      $((New-Item -ItemType Directory -Force -Path $ListDir).FullName)" }
Write-Ok "编码方式    $Encoder  (cq=$Cq crf=$Crf preset=$Preset)"
$rateDescription = "码率控制    $RateControl"
if ($TargetSize) { $rateDescription += "  目标体积 $TargetSize（自动换算码率，两遍）" }
elseif ($Bitrate) { $rateDescription += "  目标码率 $Bitrate" }
Write-Ok $rateDescription
Write-Ok "检测参数    阈值=$VadThreshold 补边=${VadPadMs}ms 合并=${VadMinSilenceMs}ms 最短=${VadMinSpeechMs}ms"
if ($DryRun) { Write-Warn '试运行模式：不会生成任何成品。' }
Write-Host ''

# ---------------------------------------------------------------- 收集源文件

$isMediaFile = {
    param($Item)
    $MediaExtensions -contains $Item.Extension.ToLower()
}

$allFiles = if ((Get-Item -LiteralPath $source).PSIsContainer) {
    Get-ChildItem -LiteralPath $source -Recurse -File -Force
}
else {
    Get-Item -LiteralPath $source
}

$runnable = @(
    foreach ($item in ($allFiles | Sort-Object FullName)) {
        # 只收媒体文件，顺带跳过隐藏项、链接、空文件和已生成的成品。
        if (-not (& $isMediaFile $item)) { continue }
        if ($item.Name.StartsWith('.')) { continue }
        if ($item.LinkType) { continue }
        if ($item.Length -eq 0) { continue }
        if ($item.BaseName.ToLower().EndsWith('_trimmed')) {
            Write-Warn "跳过已生成文件：$($item.Name)"
            continue
        }
        $item
    }
)

if ($runnable.Count -eq 0) {
    Write-Bad '没有找到可处理的媒体文件。'
    exit 1
}

Write-Step "共 $($runnable.Count) 集待处理"
Write-Host ''

# ---------------------------------------------------------------- 逐集处理

$resolvedOutputDir = if ($OutputDir) { (New-Item -ItemType Directory -Force -Path $OutputDir).FullName } else { $null }
$resolvedListDir = if ($ListDir) { (New-Item -ItemType Directory -Force -Path $ListDir).FullName } else { $null }

$results = [System.Collections.Generic.List[object]]::new()
$index = 0
$totalStopwatch = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($file in $runnable) {
    $index++
    $relative = $file.FullName.Substring($source.Length).TrimStart('\', '/')
    $label = if ($relative) { $relative } else { $file.Name }

    if ($resolvedOutputDir) {
        $destination = Join-Path $resolvedOutputDir $relative
        $destination = Join-Path (Split-Path -Parent $destination) "$($file.BaseName)_trimmed$($file.Extension)"
    }
    else {
        $destination = Join-Path $file.DirectoryName "$($file.BaseName)_trimmed$($file.Extension)"
    }

    Write-Host ("[{0}/{1}] {2}" -f $index, $runnable.Count, $label) -ForegroundColor White

    $arguments = @(
        $TrimScript, $file.FullName, $destination,
        '--encoder', $Encoder,
        '--cq', $Cq,
        '--crf', $Crf,
        '--preset', $Preset,
        '--vad-threshold', $VadThreshold,
        '--vad-pad-ms', $VadPadMs,
        '--vad-min-silence-ms', $VadMinSilenceMs,
        '--vad-min-speech-ms', $VadMinSpeechMs
    )
    if ($TargetSize) {
        $arguments += @('--target-size', $TargetSize)
        if ($RateControl -ne 'cq') { $arguments += @('--rate-control', $RateControl) }
    }
    else {
        $arguments += @('--rate-control', $RateControl)
        if ($Bitrate) { $arguments += @('--bitrate', $Bitrate) }
    }
    if ($resolvedListDir) { $arguments += @('--list-dir', $resolvedListDir) }
    if ($Overwrite) { $arguments += '--overwrite' }
    if ($DryRun) { $arguments += '--dry-run' }

    # 原生命令的 stderr 在 Stop 模式下会被当成错误中断，这里临时放宽。
    $stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
    $ErrorActionPreference = 'Continue'
    $output = & $python @arguments 2>&1
    $exitCode = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    $stopwatch.Stop()

    $output = @($output | ForEach-Object { $_.ToString() })

    $kept = $null
    $predicted = $null
    $note = $null
    foreach ($line in $output) {
        if ($line -match 'Keeping\s+\d+\s+speech range\(s\),\s+([\d.]+)\s+seconds') {
            $kept = [double]$Matches[1]
        }
        elseif ($line -match 'predicting\s+([\d.]+)\s+MB') {
            $predicted = [double]$Matches[1]
        }
        elseif ($line -match '\(\+([\d.]+)s that holds no speech') {
            $note = "关键帧对齐多留 $($Matches[1])s"
        }
        elseif ($line -match 'No speech was detected') {
            $note = '未检测到语音'
        }
    }

    $sizeMb = $null
    if (-not $DryRun -and (Test-Path -LiteralPath $destination)) {
        $sizeMb = [math]::Round((Get-Item -LiteralPath $destination).Length / 1MB, 1)
    }

    $succeeded = ($exitCode -eq 0)
    $row = [pscustomobject]@{
        集     = $label
        保留秒 = if ($null -ne $kept) { [math]::Round($kept, 1) } else { $null }
        预计MB = $predicted
        体积MB = $sizeMb
        耗时秒 = [math]::Round($stopwatch.Elapsed.TotalSeconds, 1)
        状态   = if ($succeeded) { '成功' } else { '失败' }
    }
    $results.Add($row)

    if ($succeeded) {
        $detail = "耗时 $($row.耗时秒)s"
        if ($null -ne $kept) { $detail = "保留 $($row.保留秒)s，$detail" }
        if ($null -ne $predicted -and $null -ne $sizeMb) {
            $detail += "，$sizeMb MB（目标 $predicted MB，偏差 $([math]::Round(($sizeMb - $predicted) / $predicted * 100, 1))%）"
        }
        elseif ($null -ne $sizeMb) { $detail += "，$sizeMb MB" }
        if ($note) { $detail += "（$note）" }
        Write-Ok $detail
    }
    else {
        $reason = ($output | Where-Object { $_ -match 'Failed:|Error|Traceback' } | Select-Object -Last 1)
        Write-Bad "失败（退出码 $exitCode）：$reason"
        $row | Add-Member -NotePropertyName 错误 -NotePropertyValue $reason
    }
    Write-Host ''
}

$totalStopwatch.Stop()

# ---------------------------------------------------------------- 汇总

$failed = @($results | Where-Object { $_.状态 -eq '失败' })
$keptTotal = ($results | Where-Object { $null -ne $_.保留秒 } | Measure-Object -Property 保留秒 -Sum).Sum

Write-Step '汇总'
$results | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

if ($null -ne $keptTotal) {
    Write-Ok ("累计保留 {0:N1} 秒（约 {1:N1} 分钟）" -f $keptTotal, ($keptTotal / 60))
}
Write-Ok ("总耗时 {0:N1} 秒，成功 {1} 集，失败 {2} 集" -f $totalStopwatch.Elapsed.TotalSeconds,
    ($results.Count - $failed.Count), $failed.Count)

if ($failed.Count -gt 0) {
    Write-Host ''
    Write-Bad '失败的集：'
    $failed | ForEach-Object { Write-Bad "  $($_.集) —— $($_.错误)" }
    exit 1
}