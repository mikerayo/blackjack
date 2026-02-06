@echo off
REM Quick start examples for ML Blackjack

echo ============================================
echo ML BLACKJACK - Quick Start Examples
echo ============================================
echo.

echo Choose an option:
echo 1. Test the game engine (5 random games)
echo 2. Train agent (quick test - 1,000 episodes)
echo 3. Train agent (full training - 100,000 episodes)
echo 4. Evaluate model (needs a trained model first)
echo 5. Run unit tests
echo 6. Exit
echo.

set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" (
    echo.
    echo Testing game engine...
    python src\main.py --mode test
) else if "%choice%"=="2" (
    echo.
    echo Quick training test (1,000 episodes)...
    python src\main.py --mode train --episodes 1000 --log-frequency 100 --save-frequency 500
) else if "%choice%"=="3" (
    echo.
    echo Starting full training (100,000 episodes)...
    echo This will take a while. Grab a coffee! ☕
    python src\main.py --mode train --episodes 100000 --log-frequency 1000 --save-frequency 10000 --visualize
) else if "%choice%"=="4" (
    echo.
    echo Available models:
    dir /B models\*.pt
    echo.
    set /p model_path="Enter model filename: "
    echo Evaluating model...
    python src\main.py --mode evaluate --episodes 10000 --model-path models\%model_path%
) else if "%choice%"=="5" (
    echo.
    echo Running tests...
    python tests\test_env.py
    echo.
    python tests\test_game.py
) else if "%choice%"=="6" (
    exit
) else (
    echo Invalid choice. Exiting.
)

echo.
pause
