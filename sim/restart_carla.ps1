# Runs in: Windows PowerShell. Fresh CARLA server before each scenario (it stops answering after a run when RAM is tight).
#   powershell -ExecutionPolicy Bypass -File D:\Project_ai\sim\restart_carla.ps1
Stop-Process -Name CarlaUE4-Win64-Shipping,CarlaUE4 -Force -ErrorAction SilentlyContinue
Start-Sleep 8
Start-Process -FilePath "D:\CARLA_0.9.15\CarlaUE4.exe" -ArgumentList "-quality-level=Low","-RenderOffScreen","-fps=20" -WindowStyle Hidden
$ok=$false; for($i=0;$i -lt 30;$i++){ Start-Sleep 5; if(Get-NetTCPConnection -LocalPort 2000 -State Listen -ErrorAction SilentlyContinue){$ok=$true;break} }
Start-Sleep 10
"listening=$ok avail_MB=" + (Get-Counter '\Memory\Available MBytes').CounterSamples.CookedValue
