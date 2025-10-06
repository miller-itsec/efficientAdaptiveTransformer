@echo off
:: This batch file compiles a LaTeX document with a bibliography.
:: It runs pdflatex -> bibtex -> pdflatex -> pdflatex.
:: It now redirects all output to compile.log for debugging.

setlocal
set TEX_FILE=%~1
set BASE_NAME=%~n1
set LOG_FILE=compile.log

if not defined TEX_FILE (
    echo Error: No .tex file provided.
    echo Usage: Drag your .tex file onto this script.
    pause
    exit /b 1
)

:: Clear the old log file for a fresh start
if exist "%LOG_FILE%" del "%LOG_FILE%"

echo ======================================================
echo  Compiling: %TEX_FILE%
echo  (All output is being redirected to %LOG_FILE%)
echo ======================================================

:: Step 1: Initial pdflatex run to generate .aux file
echo [1/4] Running pdflatex...
pdflatex -interaction=nonstopmode "%BASE_NAME%.tex" >> "%LOG_FILE%" 2>>&1
if errorlevel 1 (echo An error occurred in pdflatex. See %LOG_FILE%. & pause & exit /b)

:: Step 2: Run bibtex to process citations
echo [2/4] Running bibtex...
bibtex "%BASE_NAME%" >> "%LOG_FILE%" 2>>&1
if errorlevel 1 (echo An error occurred in bibtex. See %LOG_FILE%. & pause & exit /b)

:: Step 3: Second pdflatex run to include the bibliography
echo [3/4] Running pdflatex again...
pdflatex -interaction=nonstopmode "%BASE_NAME%.tex" >> "%LOG_FILE%" 2>>&1
if errorlevel 1 (echo An error occurred in pdflatex. See %LOG_FILE%. & pause & exit /b)

:: Step 4: Final pdflatex run to fix cross-references
echo [4/4] Final pdflatex run...
pdflatex -interaction=nonstopmode "%BASE_NAME%.tex" >> "%LOG_FILE%" 2>>&1
if errorlevel 1 (echo An error occurred in pdflatex. See %LOG_FILE%. & pause & exit /b)

echo ======================================================
echo  Compilation finished.
echo  A detailed log has been saved to: %LOG_FILE%
echo  PDF generated: %BASE_NAME%.pdf
echo ======================================================

pause
endlocal