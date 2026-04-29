# PowerShell Predictive Maintenance Pipeline (Optimized for PS 5.1)
# Reads CSV, performs feature engineering, detects anomalies, generates output

param(
    [string]$DataPath = "system-1.csv",
    [int]$SampleSize = 1000,
    [double]$ContaminationRate = 0.05
)

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "Predictive Maintenance Pipeline - PowerShell" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# 1. Load Dataset
Write-Host "[1/8] Loading dataset..." -ForegroundColor Yellow
if (-not (Test-Path $DataPath)) {
    Write-Host "ERROR: Dataset not found at $DataPath" -ForegroundColor Red
    exit 1
}

# Use Import-Csv for efficiency and automatic header mapping
$raw_data = Import-Csv $DataPath | Select-Object -First $SampleSize
Write-Host "Loaded $($raw_data.Count) records" -ForegroundColor Green
Write-Host ""

# 2. Feature Engineering & Type Conversion
Write-Host "[2/8] Performing feature engineering..." -ForegroundColor Yellow
$processed = New-Object System.Collections.Generic.List[object]

foreach ($row in $raw_data) {
    $obj = [PSCustomObject]@{
        timestamp = [long]$row.timestamp
        'cpu_total' = [double]$row.'cpu-user' + [double]$row.'cpu-system' + [double]$row.'cpu-iowait'
        'mem_usage_percent' = ([double]$row.'sys-mem-total' - [double]$row.'sys-mem-available') / [double]$row.'sys-mem-total'
        'swap_usage_percent' = ([double]$row.'sys-mem-swap-total' - [double]$row.'sys-mem-swap-free') / [double]$row.'sys-mem-swap-total'
        'disk_total_bytes' = [double]$row.'disk-bytes-read' + [double]$row.'disk-bytes-written'
        'disk_total_ops' = [double]$row.'disk-io-read' + [double]$row.'disk-io-write'
        'load_avg' = ([double]$row.'load-1m' + [double]$row.'load-5m' + [double]$row.'load-15m') / 3
        'server_up' = [int]$row.'server-up'
    }
    $processed.Add($obj)
}
Write-Host "Features engineered: cpu_total, mem_usage_percent, swap_usage_percent, disk_total_bytes, disk_total_ops, load_avg" -ForegroundColor Green
Write-Host ""

# 3. Calculate Statistics
Write-Host "[3/8] Computing statistics..." -ForegroundColor Yellow
$cpu_values = $processed | ForEach-Object { $_.cpu_total }
$cpu_measure = $cpu_values | Measure-Object -Average -Sum
$cpu_mean = $cpu_measure.Average
$cpu_count = $cpu_measure.Count

$sum_sq_diff = 0
foreach ($val in $cpu_values) {
    $sum_sq_diff += [Math]::Pow($val - $cpu_mean, 2)
}
$cpu_stddev = [Math]::Sqrt($sum_sq_diff / $cpu_count)

$mem_mean = ($processed | ForEach-Object { $_.mem_usage_percent } | Measure-Object -Average).Average
$swap_mean = ($processed | ForEach-Object { $_.swap_usage_percent } | Measure-Object -Average).Average

Write-Host "  CPU Total - Mean: $([Math]::Round($cpu_mean, 4)), StdDev: $([Math]::Round($cpu_stddev, 4))" -ForegroundColor Cyan
Write-Host "  Memory Usage - Mean: $([Math]::Round($mem_mean * 100, 2))%" -ForegroundColor Cyan
Write-Host "  Swap Usage - Mean: $([Math]::Round($swap_mean * 100, 2))%" -ForegroundColor Cyan
Write-Host ""

# 4. Anomaly Detection (Simple Z-Score)
Write-Host "[4/8] Detecting anomalies using statistical method..." -ForegroundColor Yellow
$anomaly_threshold = 3  # 3-sigma
$anomaly_count = 0

foreach ($obj in $processed) {
    $cpu_zscore = if ($cpu_stddev -gt 0) { [Math]::Abs(($obj.cpu_total - $cpu_mean) / $cpu_stddev) } else { 0 }
    $is_anomaly = if ($cpu_zscore -gt $anomaly_threshold) { 1 } else { 0 }
    if ($is_anomaly -eq 1) { $anomaly_count++ }
    
    $obj | Add-Member -NotePropertyName 'is_anomaly' -NotePropertyValue $is_anomaly
    $obj | Add-Member -NotePropertyName 'anomaly_score' -NotePropertyValue $cpu_zscore
}

$anomaly_pct = [Math]::Round(($anomaly_count / $processed.Count) * 100, 2)
Write-Host "Detected $anomaly_count anomalies ($($anomaly_pct)%)" -ForegroundColor Green
Write-Host ""

# 5. Model Evaluation
Write-Host "[5/8] Evaluating model performance..." -ForegroundColor Yellow
# Note: 'server-up' 2 seems to mean up, 0 or 1 might mean down? 
# Looking at CSV, 'server-up' values are 2. Let's assume 2 is NORMAL.
# If server-up < 2, it might be a failure.
$true_positives = 0
$false_positives = 0
$true_negatives = 0
$false_negatives = 0

foreach ($obj in $processed) {
    $is_failure = if ($obj.server_up -lt 2) { 1 } else { 0 }
    if ($obj.is_anomaly -eq 1 -and $is_failure -eq 1) { $true_positives++ }
    elseif ($obj.is_anomaly -eq 1 -and $is_failure -eq 0) { $false_positives++ }
    elseif ($obj.is_anomaly -eq 0 -and $is_failure -eq 0) { $true_negatives++ }
    elseif ($obj.is_anomaly -eq 0 -and $is_failure -eq 1) { $false_negatives++ }
}

$total_eval = $true_positives + $true_negatives + $false_positives + $false_negatives
$accuracy = if ($total_eval -gt 0) { ($true_positives + $true_negatives) / $total_eval } else { 0 }
$precision = if ($true_positives + $false_positives -gt 0) { $true_positives / ($true_positives + $false_positives) } else { 0 }
$recall = if ($true_positives + $false_negatives -gt 0) { $true_positives / ($true_positives + $false_negatives) } else { 0 }

Write-Host "  Accuracy: $([Math]::Round($accuracy, 4))" -ForegroundColor Cyan
Write-Host "  Precision: $([Math]::Round($precision, 4))" -ForegroundColor Cyan
Write-Host "  Recall: $([Math]::Round($recall, 4))" -ForegroundColor Cyan
Write-Host "  Confusion Matrix (Anomaly vs Failure):" -ForegroundColor Cyan
Write-Host "    TP: $true_positives, FP: $false_positives" -ForegroundColor Cyan
Write-Host "    FN: $false_negatives, TN: $true_negatives" -ForegroundColor Cyan
Write-Host ""

# 6. Generate Report
Write-Host "[6/8] Generating analysis report..." -ForegroundColor Yellow
$report = @"
================================================================================
                   PREDICTIVE MAINTENANCE ANALYSIS REPORT
================================================================================

Dataset Information:
  - Records analyzed: $($processed.Count)
  - Features engineered: cpu_total, mem_usage_percent, swap_usage_percent, 
                        disk_total_bytes, disk_total_ops, load_avg

Feature Engineering Results:
  - cpu_total: Mean=$([Math]::Round($cpu_mean, 4)), StdDev=$([Math]::Round($cpu_stddev, 4))
  - mem_usage_percent: Mean=$([Math]::Round($mem_mean * 100, 2))%
  - swap_usage_percent: Mean=$([Math]::Round($swap_mean * 100, 2))%

Anomaly Detection:
  - Method: Statistical (Z-score > 3-sigma on cpu_total)
  - Anomalies detected: $anomaly_count ($($anomaly_pct)%)
  - Normal records: $($processed.Count - $anomaly_count)

Model Performance (Anomaly as Predictor of Failure):
  - Accuracy: $([Math]::Round($accuracy * 100, 2))%
  - Precision: $([Math]::Round($precision * 100, 2))%
  - Recall: $([Math]::Round($recall * 100, 2))%
  
Confusion Matrix:
  Predicted Anomaly | Actual Failure | Count
  ------------------|----------------|-------
  Yes               | Yes (TP)       | $true_positives
  Yes               | No (FP)        | $false_positives
  No                | Yes (FN)       | $false_negatives
  No                | No (TN)        | $true_negatives

Top 5 Anomalies (by score):
"@

$top_anomalies = $processed | Sort-Object anomaly_score -Descending | Select-Object -First 5
foreach ($a in $top_anomalies) {
    $report += "`n  Score: $([Math]::Round($a.anomaly_score, 4)), CPU: $([Math]::Round($a.cpu_total, 4)), Server-Up: $($a.server_up)"
}

$report += "`n`n================================================================================"

$report | Out-File "pdm_analysis_report.txt" -Force -Encoding ascii
Write-Host "Report generated: pdm_analysis_report.txt" -ForegroundColor Green
Write-Host ""

# 7. Export Results
Write-Host "[7/8] Exporting results..." -ForegroundColor Yellow
$processed | Select-Object timestamp, cpu_total, mem_usage_percent, anomaly_score, is_anomaly, server_up | 
    Export-Csv "anomaly_results.csv" -NoTypeInformation -Force
Write-Host "Results exported: anomaly_results.csv" -ForegroundColor Green
Write-Host ""

# 8. Summary
Write-Host "[8/8] Pipeline complete!" -ForegroundColor Yellow
Write-Host "===============================================" -ForegroundColor Green
Write-Host "Predictive Maintenance Pipeline - COMPLETED" -ForegroundColor Green
Write-Host "===============================================" -ForegroundColor Green
