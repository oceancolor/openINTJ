# Renders the product name "openINTJ" into resources/icon.png (1024x1024).
# The icon is the name — no extra mark.
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Drawing

$out = Join-Path $PSScriptRoot "..\resources\icon.png"
$size = 1024
$bg = [System.Drawing.Color]::FromArgb(255, 30, 30, 46)     # desktop chrome #1e1e2e
$muted = [System.Drawing.Color]::FromArgb(255, 166, 173, 200)
$fg = [System.Drawing.Color]::FromArgb(255, 205, 214, 244)  # desktop text

$bmp = New-Object System.Drawing.Bitmap $size, $size, ([System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.Clear($bg)

$familyName = "Segoe UI"
$unit = [System.Drawing.GraphicsUnit]::Pixel
$fmt = New-Object System.Drawing.StringFormat
$fmt.Alignment = [System.Drawing.StringAlignment]::Center
$fmt.LineAlignment = [System.Drawing.StringAlignment]::Center

$openBrush = New-Object System.Drawing.SolidBrush $muted
$intjBrush = New-Object System.Drawing.SolidBrush $fg
$openFont = New-Object System.Drawing.Font($familyName, [float]168, [System.Drawing.FontStyle]::Regular, $unit)
$intjFont = New-Object System.Drawing.Font($familyName, [float]300, [System.Drawing.FontStyle]::Bold, $unit)

$openRect = New-Object System.Drawing.RectangleF 48, 220, 928, 220
$intjRect = New-Object System.Drawing.RectangleF 48, 420, 928, 380
$g.DrawString("open", $openFont, $openBrush, $openRect, $fmt)
$g.DrawString("INTJ", $intjFont, $intjBrush, $intjRect, $fmt)

$bmp.Save($out, [System.Drawing.Imaging.ImageFormat]::Png)

$openFont.Dispose(); $intjFont.Dispose()
$openBrush.Dispose(); $intjBrush.Dispose()
$fmt.Dispose(); $g.Dispose(); $bmp.Dispose()
Write-Output "wrote $out"
