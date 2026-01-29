@echo off
setlocal enabledelayedexpansion

echo ========================================
echo     Build Script for CUDA Miner
echo     Windows Version
echo ========================================

REM Check if NVCC is available
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: NVCC not found in PATH!
    echo Please install CUDA Toolkit and add it to PATH.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

REM Step 1: Try to detect GPU compute capability
echo.
echo --- Detecting GPU Compute Capability ---
for /f "tokens=*" %%i in ('nvidia-smi --query-gpu^=compute_cap --format^=csv,noheader 2^>nul') do (
    set GPU_CAPABILITY=%%i
)

if "!GPU_CAPABILITY!"=="" (
    echo Warning: Could not detect GPU compute capability automatically.
    echo Please enter compute capability manually (e.g., 8.6 for RTX 30 series):
    set /p GPU_CAPABILITY=
    
    if "!GPU_CAPABILITY!"=="" (
        echo Using default compute capability: 5.0
        set GPU_CAPABILITY=5.0
    )
)

REM Format for NVCC (e.g., 8.6 -> sm_86)
set GPU_CAPABILITY=!GPU_CAPABILITY: =!
set ARCH_FLAG=sm_!GPU_CAPABILITY:.=!

echo Detected GPU Compute Capability: !GPU_CAPABILITY!
echo Using NVCC architecture flag: -arch=!ARCH_FLAG!

REM Step 2: Check if CUDA files exist
if not exist "rr.cu" (
    echo Error: rr.cu not found!
    pause
    exit /b 1
)

if not exist "liblooper.cu" (
    echo Error: liblooper.cu not found!
    pause
    exit /b 1
)

REM Step 3: Compile rr.cu to rr.dll
echo.
echo --- Compiling rr.cu to rr.dll ---
nvcc -o rr.dll ^
     -O3 ^
     --shared ^
     -Xcompiler /MD ^
     --use_fast_math ^
     -arch=!ARCH_FLAG! ^
     rr.cu

if %errorlevel% neq 0 (
    echo Error: Compilation of rr.dll failed!
    pause
    exit /b 1
)

echo ✓ Successfully compiled rr.dll

REM Step 4: Compile liblooper.cu to liblooper.dll
echo.
echo --- Compiling liblooper.cu to liblooper.dll ---
nvcc -o liblooper.dll ^
     -O3 ^
     --shared ^
     -Xcompiler /MD ^
     --use_fast_math ^
     -arch=!ARCH_FLAG! ^
     liblooper.cu

if %errorlevel% neq 0 (
    echo Error: Compilation of liblooper.dll failed!
    pause
    exit /b 1
)

echo ✓ Successfully compiled liblooper.dll

REM Step 5: Copy Python files if needed
echo.
echo --- Checking Python files ---
if exist "looper.py" (
    echo ✓ looper.py found
) else (
    echo Warning: looper.py not found in current directory
)

if exist "rrnonce.py" (
    echo ✓ rrnonce.py found
) else (
    echo Warning: rrnonce.py not found in current directory
)

echo.
echo ========================================
echo     BUILD COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo Files generated:
echo   - rr.dll           (CUDA miner library)
echo   - liblooper.dll    (Looper CUDA library)
echo.
echo You can now run:
echo   python test.py or python3 test.py
echo.
pause
