
# Parallel Image Blending

## Overview

This repository contains the parallel implementation of image blending using various parallel computing techniques, including OpenMP, POSIX threads (pthread), and SIMD (Intel MMX SIMD). The project showcases how leveraging parallelism can significantly improve the performance of image blending tasks by distributing the workload across multiple threads and processing units.

## Table of Contents

1. [Introduction](#introduction)
2. [Project Structure](#project-structure)
3. [Implementation Details](#implementation-details)
   - [OpenMP](#openmp)
   - [POSIX Threads (pthread)](#posix-threads-pthread)
   - [SIMD](#simd)
4. [Results](#results)
5. [Usage](#usage)
6. [License](#license)

## Introduction

The goal of this project is to blend two images using parallel computing techniques, which is essential in various image processing and computer vision tasks. This project demonstrates how parallelism can accelerate the image blending process by distributing the computational workload across multiple threads or vectorized instructions. The project includes implementations using:

- OpenMP (Open Multi-Processing)
- POSIX Threads (pthread)
- SIMD (Single Instruction, Multiple Data) using Intel MMX SIMD features

## Project Structure

The project is organized into the following directories, each containing the implementation of the image blending algorithm using a different parallelization technique:

- `OpenMP/`: Contains the image blending implementation using OpenMP.
- `Pthread/`: Contains the image blending implementation using POSIX threads.
- `SIMD/`: Contains the image blending implementation using Intel MMX SIMD instructions.

Each directory contains a `main.cpp` file along with necessary dependencies for that parallelization technique.

## Implementation Details

### OpenMP

The OpenMP implementation utilizes the OpenMP library to parallelize the image blending process across multiple CPU cores. OpenMP simplifies the parallelization process by allowing easy annotation of loops to distribute work across available threads.

**File:** `OpenMP/main.cpp`

### POSIX Threads (pthread)

The pthread implementation uses POSIX threads to manually manage thread creation, workload distribution, and synchronization. This method provides fine-grained control over how threads are used during the image blending operation.

**File:** `Pthread/main.cpp`

### SIMD

The SIMD implementation uses Intel's MMX SIMD instructions to blend the images in parallel at the hardware level. SIMD allows multiple data points to be processed simultaneously within a single CPU core, making it highly efficient for image processing tasks like blending.

**File:** `SIMD/main.cpp`

## Results

The performance of each implementation is evaluated by measuring the execution time and speedup achieved compared to the serial execution baseline. Results are printed to the console, showing the runtime of each method and the resulting blended image.

## Usage

To compile and run the programs, navigate to the respective directory and use the following commands:

### OpenMP

```sh
g++ -fopenmp -o imageBlenderOpenMP main.cpp `pkg-config --cflags --libs opencv4`
./imageBlenderOpenMP
```

### POSIX Threads (pthread)

```sh
g++ -lpthread -o imageBlenderPthread main.cpp `pkg-config --cflags --libs opencv4`
./imageBlenderPthread
```

### SIMD

```sh
g++ -o imageBlenderSIMD main.cpp `pkg-config --cflags --libs opencv4`
./imageBlenderSIMD
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
