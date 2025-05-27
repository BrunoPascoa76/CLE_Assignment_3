# CLE Assignment 3 Overview

## Introduction

This report is an overview of the work done in converting a canny edge detection script to use CUDA cores for better efficiency.

## Code overview

### Gaussian filter

Note: the gaussian_filter_cuda() function itself runs on the CPU, but it's main purpose is to launch gaussian-related kernels (while keeping the code separated for readability)

#### Kernel calculation

#### Convolution (Gaussian filter)

For the convolution kernel (used both in the gaussian and sobel filters), we utilized the same formula as the CPU version. However, by mapping each thread to a pixel, we were able to remove the 2 outermost loops (only needing 2 loops for applying the kernel to the neighbors).  
The only additional precaution needed was to have pixels in which the kernel would go out of bounds return immediately, leaving them untouched.

#### min_max

In order to normalize later, we would need to gather the minimum and maximum brightness values across the whole image.  
However, following the naive approach and have each thread compare its value, that would lead to a lot of contention, so we first calculate the local minimum/maximum for each block (using shared memory and parallel reduction) and only compare those values to the current maximum/minimum, greatly reducing wasted time.

#### normalize

In the actual normalization, we simply had the thread get the coordinates of the corresponding pixel (returning immediately if it was a "border pixel") and apply the normalization directly (using the same formula as the CPU version).

### Sobel filter

#### Convolutions

For the sobel filter, while the convolution kernel itself was the same as the one used for the gaussian filter. However, instead of being applyed once, the sobel filter uses 2 convolutions: one using a horizontal kernel (Gx) and another using a vertical kernel (Gy)

#### Merge kernels

For merging the actual kernels, I simply applied the same formula as the CPU version, only removing the 2 for loops by mapping each thread to a pixel.

### Non maximum suppression

Much like other functions before, this one follows the same formulas and logic as the CPU version, only removing the for loops to iterate every pixel by mapping each thread to a pixel.