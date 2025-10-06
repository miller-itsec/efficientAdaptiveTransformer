@echo off
REM ============================================================
REM  Efficient Adaptive Transformer — arXiv Submission Builder
REM  Creates a clean ZIP of LaTeX + figures for arXiv upload
REM ============================================================

setlocal
set PROJECT=efficient_adaptive_transformer
set SRC_DIR=%~dp0
set OUT_DIR=%SRC_DIR%\arxiv_package
set ZIP_NAME=%PROJECT%_source.zip

echo.
echo [*] Preparing arXiv submission package...
echo     Source: %SRC_DIR%
echo     Output: %OUT_DIR%
echo.

REM --- 1. Clean any previous build ---
if exist "%OUT_DIR%" (
    echo [*] Removing old package folder...
    rmdir /s /q "%OUT_DIR%"
)
mkdir "%OUT_DIR%"

REM --- 2. Recreate project directory structure for TeX ---
echo [*] Recreating required directory structure (results/plots, results/tables)...
mkdir "%OUT_DIR%\results"
mkdir "%OUT_DIR%\results\plots"
mkdir "%OUT_DIR%\results\tables"

REM --- 3. Copy core LaTeX sources ---
echo [*] Copying LaTeX sources...
copy "%SRC_DIR%\%PROJECT%.tex" "%OUT_DIR%" >nul
if exist "%SRC_DIR%\references.bib" copy "%SRC_DIR%\references.bib" "%OUT_DIR%" >nul

REM --- 4. Copy required figure PDFs ---
echo [*] Copying figures into results\plots\...
for %%F in (
    frontier_sst2.pdf
    frontier_qqp.pdf
    frontier_mnli.pdf
    ablation_sst2.pdf
    retention_vs_length.pdf
    exit_distribution.pdf
) do (
    if exist "%SRC_DIR%\results\plots\%%F" (
        echo     ✓ Found %%F
        copy "%SRC_DIR%\results\plots\%%F" "%OUT_DIR%\results\plots\" >nul
    ) else (
        echo     ! Missing %%F ^(skipped^) - This might cause a LaTeX error.
    )
)

REM --- 5. Copy required table CSVs ---
echo [*] Copying tables into results\tables\...
for %%F in (
    summary_sst2.csv
    summary_qqp.csv
    summary_mnli.csv
    summary_ablation_sst2.csv
) do (
    if exist "%SRC_DIR%\results\tables\%%F" (
        echo     ✓ Found %%F
        copy "%SRC_DIR%\results\tables\%%F" "%OUT_DIR%\results\tables\" >nul
    ) else (
        echo     ! Note: Optional table %%F not found ^(skipped^).
    )
)

REM --- 6. Create README for arXiv ---
echo [*] Creating README_arxiv.txt...
(
echo Efficient Adaptive Transformer — arXiv Source Package
echo -----------------------------------------------------
echo This archive contains the LaTeX source, figures, and tables
echo required to compile the paper.
echo.
echo Contents:
echo   - efficient_adaptive_transformer.tex
echo   - references.bib
echo   - results/plots/ (PDF figures)
echo   - results/tables/ (CSV data for tables)
echo.
echo This package is structured to compile directly without any
echo path modifications to the .tex file.
) > "%OUT_DIR%\README_arxiv.txt"

REM --- 7. Create ZIP archive ---
cd "%OUT_DIR%"
if exist "%SRC_DIR%\%ZIP_NAME%" del "%SRC_DIR%\%ZIP_NAME%"
powershell -Command "Compress-Archive -Path * -DestinationPath '%SRC_DIR%\%ZIP_NAME%' -Force"

if exist "%SRC_DIR%\%ZIP_NAME%" (
    echo [✓] Created %ZIP_NAME% successfully!
    echo     Upload this ZIP to arXiv as a LaTeX source submission.
) else (
    echo [✗] ZIP creation failed. Check permissions or paths.
)

echo.
pause
endlocal