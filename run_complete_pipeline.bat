@echo off
echo ============================================================
echo SMART ENERGY ANALYSIS - COMPLETE PIPELINE
echo ============================================================
echo.

echo [1/5] Creating necessary directories...
if not exist data mkdir data
if not exist data\processed mkdir data\processed
if not exist models mkdir models
if not exist reports mkdir reports
if not exist reports\results mkdir reports\results
if not exist reports\figures mkdir reports\figures
echo    Done!
echo.

echo [2/5] Running data preprocessing...
echo    This may take 2-5 minutes...
python src\data_preprocessing.py
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR: Data preprocessing failed!
    echo    Please check if HomeC_augmented.csv exists.
    pause
    exit /b 1
)
echo    Done!
echo.

echo [3/5] Running feature engineering...
python src\feature_engineering.py
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR: Feature engineering failed!
    pause
    exit /b 1
)
echo    Done!
echo.

echo [4/5] Training baseline model...
python src\baseline_model.py
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR: Baseline model training failed!
    pause
    exit /b 1
)
echo    Done!
echo.

echo [5/5] Generating energy suggestions...
python src\suggestions.py
if %ERRORLEVEL% NEQ 0 (
    echo    ERROR: Suggestions generation failed!
    pause
    exit /b 1
)
echo    Done!
echo.

echo ============================================================
echo SUCCESS! All modules completed successfully!
echo ============================================================
echo.
echo You can now run the web application:
echo    python app.py
echo.
echo Then open your browser to: http://localhost:5000
echo ============================================================
pause
