@echo off
REM ML Blackjack - Advanced System Quick Start

echo ================================================================================
echo ML BLACKJACK - ADVANCED SYSTEM
echo ================================================================================
echo.
echo Choose an option:
echo.
echo [1] Test Advanced Features (Quick - 30 seconds)
echo [2] Evaluate All Expert Strategies (10K episodes each - ~5 min)
echo [3] Compare Consensus Systems (10K episodes - ~2 min)
echo [4] Compare Betting Systems (10K episodes - ~2 min)
echo [5] Start MASSIVE Training (5M episodes - 6-12 hours)
echo [6] Quick Training Test (100K episodes - ~30 min)
echo [7] Resume from Checkpoint
echo [8] Exit
echo.

set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" (
    echo.
    echo Testing advanced features...
    python test_advanced.py
) else if "%choice%"=="2" (
    echo.
    echo Evaluating ALL expert strategies...
    echo This will take a few minutes...
    python evaluate_strategies.py --episodes 10000 --type strategies
) else if "%choice%"=="3" (
    echo.
    echo Comparing consensus systems...
    python evaluate_strategies.py --episodes 10000 --type consensus
) else if "%choice%"=="4" (
    echo.
    echo Comparing betting systems...
    python evaluate_strategies.py --episodes 10000 --type betting
) else if "%choice%"=="5" (
    echo.
    echo STARTING MASSIVE TRAINING - 5 MILLION EPISODES
    echo This will take 6-12 hours. Make sure you have time!
    echo.
    set /p confirm="Continue? (y/n): "
    if /i "%confirm%"=="y" (
        echo.
        echo Starting training...
        python train_massive.py --episodes 5000000 --use-consensus --use-variable-betting
    ) else (
        echo Cancelled.
    )
) else if "%choice%"=="6" (
    echo.
    echo Quick training test - 100K episodes...
    python train_massive.py --episodes 100000 --use-consensus --consensus-type hybrid --checkpoint-interval 25000 --log-interval 5000
) else if "%choice%"=="7" (
    echo.
    echo Available checkpoints:
    dir /B models\checkpoints\*.pt
    echo.
    set /p checkpoint="Enter checkpoint filename: "
    echo Resuming from checkpoint...
    python train_massive.py --episodes 5000000 --resume models\checkpoints\%checkpoint%
) else if "%choice%"=="8" (
    exit
) else (
    echo Invalid choice.
)

echo.
pause
