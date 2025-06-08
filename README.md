# B3Search

B3Search: A Bounded Branchless Binary Search algorithm optimized for GPU performance

## Overview

**B3Search** is a high-performance, branchless binary search algorithm designed for GPU execution. It is particularly optimized for use with [ILGPU](https://github.com/m4rs-mt/ILGPU), enabling efficient parallel searches on modern GPUs. The algorithm is suitable for searching sorted arrays of unsigned integers and is implemented in C# for .NET 8.

## Features

- **Branchless**: Avoids conditional branches for better performance on modern GPUs.
- **Bounded**: Prevents out-of-bounds access by clamping indices.
- **GPU-Accelerated**: Includes a kernel for batch searching using ILGPU.
- **Flexible**: Supports both 32-bit and 64-bit index ranges.
- **Easy Integration**: Provided as a .NET library for seamless use in your projects.

## Getting Started

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/download)
- [ILGPU 1.5.2](https://www.nuget.org/packages/ILGPU/1.5.2)

### Installation

Clone the repository and restore dependencies:
git clone https://github.com/DevNullx64/B3Search
cd B3Search
dotnet restore

### Usage

```csharp
using ILGPU;
using ILGPU.Runtime;
using B3Search;

using var context = Context.CreateDefault();
using var accelerator = context.CreateDefaultAccelerator();
uint[] sortedArray = { 1, 3, 5, 7, 9 };
uint[] valuesToFind = { 2, 6, 10 };
int[] indices = accelerator.GpuSearch(sortedArray, valuesToFind, out gpuTicks); // indices = [1, 3, 4]
```


## API

### `int B3Search.Search(uint[] array, uint value)`
Finds the first index where `array[index] >= value`. Returns the last index if not found.

### `int[] Accelerator.GpuSearch(uint[] array, uint[] values, out long internalTicks)`
Performs batch binary search on the GPU for each value in `values`. Returns an array of indices.

## Performance

B3Search is designed to minimize branching and maximize throughput, especially on SIMD and GPU architectures. The branchless approach reduces divergence and improves parallel efficiency.

## License

This project is licensed under the GPL-3.0 license.

## Acknowledgements

- [ILGPU](https://github.com/m4rs-mt/ILGPU) for GPU acceleration.
- Inspired by research on branchless algorithms and parallel search techniques.
