# CLE Assignment 3 Overview

## Introduction

This report is an overview of the work done in converting a canny edge detection script to use CUDA cores for better efficiency.

## Setup Instructions

## Code overview

### Gaussian filter

Note: the gaussian_filter_cuda() function itself runs on the CPU, but it's main purpose is to launch gaussian-related kernels (while keeping the code separated for readability)

#### Gaussian Kernel calculation

//change this later

- This CUDA kernel generates a 2D Gaussian filter by having each GPU thread compute one element of the n×n matrix. Each thread gets a unique linear index (idx), converts it to 2D coordinates (i,j) using division and module, then applies the Gaussian formula. 

#### Convolution (Gaussian filter)

For the convolution kernel (used both in the gaussian and sobel filters), we utilized the same formula as the CPU version. However, by mapping each thread to a pixel, we were able to remove the 2 outermost loops (only needing 2 loops for applying the kernel to the neighbors).  
The only additional precaution needed was to have pixels in which the kernel would go out of bounds return immediately, leaving them untouched.

#### min_max

In order to normalize later, we would need to gather the minimum and maximum brightness values across the whole image.  
However, following the naive approach and have each thread compare its value, that would lead to a lot of contention, so we first calculate the local minimum/maximum for each block (using shared memory and parallel reduction) and only compare those values to the current maximum/minimum, greatly reducing wasted time.

#### normalize

In the actual normalization, we simply had the thread get the coordinates of the corresponding pixel (returning immediately if it was a "border pixel") and apply the normalization directly (using the same formula as the CPU version).

### First Edges

- The First Edges Function serves as the initial edge detection pass in the Canny algorithm, identifying pixels with gradient magnitudes above the high threshold (tmax) as definitive strong edges. 

- The ***CPU version** processes pixels sequentially using nested loops, maintaining a linear index c that increments through each row while skipping border pixels by adding 2 at the end of each row. 

- The **GPU version** parallelizes this operation by having each CUDA thread handle a single pixel, calculating its position using blockIdx, blockDim, and threadIdx to derive 2D coordinates (x,y) and convert them to a linear index c = x + y * nx. 

- Both versions apply the same simple logic: if a pixel's non-maximum suppressed gradient value meets or exceeds the high threshold, it's marked as a strong edge by setting its value to MAX_BRIGHTNESS (255) in the reference image. 

### Hysteresis Edges 

- The Hysteresis Edges function connects weak edge pixels(above tmin threshold) to existing strong edges through the analysis of the 8 neighbouring pixels, creating strong edge contours.

- The **CPU version** processes pixels sequentially, checking each pixel's 8 neighbors and setting a boolean flag when changes occur, requiring the algorithm to iterate until no more weak edges can be connected to strong ones

- The **GPU version** parallelizes this process by having each thread examine one pixel simultaneously, but due to the inherently iterative nature of edge propagation, the kernel must be launched repeatedly from the host until convergence.

- To handle concurrent updates safely, the GPU implementation uses **atomicExch(changed, 1)**, instead of a simple boolean flag, preventing race conditions when multiple threads simultaneously discover they can connect weak edges to strong ones. 

### Sobel filter

#### Convolutions

For the sobel filter, while the convolution kernel itself was the same as the one used for the gaussian filter. However, instead of being applyed once, the sobel filter uses 2 convolutions: one using a horizontal kernel (Gx) and another using a vertical kernel (Gy)

#### Merge kernels

For merging the actual kernels, I simply applied the same formula as the CPU version, only removing the 2 for loops by mapping each thread to a pixel.

### Non maximum suppression

Much like other functions before, this one follows the same formulas and logic as the CPU version, only removing the for loops to iterate every pixel by mapping each thread to a pixel.


## Performance Evaluation

### Speedup calculations

Note: due to an unknown glitch (we suspect to be cache-related), the 1st runtime of a certain image may rarely be much higher than expected. To account for this, run the program twice for each image and take into account only the 2nd result.

| image  | host time (ms)  | device time (ms) | speedup | different pixels (#)  | different pixels (%)   |
|---|---|---|---|---|---|
| house.pgm   | 28.677536   | 1.551584   | 18.48x  | 0 | 0.00% |
| lake.pgm  | 32.892288   | 1.560448   | 21.08x  | 1  | 0.00%  |
| mandrill.pgm  | 35.078049  | 1.543264 | 22.73x  | 0  | 0.00%  |
| pirate.pgm | 35.022209 | 1.642752 | 21.32x | 1 | 0.00% |
| jetplane.pgm | 38.491585 | 1.554624 | 24.76x | 0 | 0.00% |
| livingroom.pgm | 32.986145 | 1.638432 | 20.13x | 0 | 0.00% |
| peppers_gray.pgm | 34.149952 | 1.615264 | 21.14x | 0 | 0.00% |
| walkbridge.pgm | 41.146976 | 1.612320 | 25.52x | 2 | 0.00% |

So, overall device times remained relatively stable throughout all images, with speedups between 18x and 26x. This difference, however, does not seem to be tied to a specific image or level of complexity, but rather just the natural variation in performance when dealing with programs with such small runtimes.

Aditionally, while float handling led to some small differences between the 2 generated images, as these were usually 2 pixels or less, we deemed it to be an acceptable difference.

### Performance breakdown

Note: This data was obtained via the command `nsys profile --stats=true ./canny` and are in nanoseconds

| Time (%) | Total Time | Instances | Average    | Minimum  | Maximum  | Name                             |
|----------|------------|-----------|------------|----------|----------|----------------------------------|
| 35.1     | 156137     | 3         | 52045.7    | 30530    | 92838    | convolution_cuda_kernel          |
| 18.4     | 81925      | 6         | 13654.2    | 12673    | 17217    | hysteresis_edges_kernel          |
| 18.1     | 80677      | 1         | 80677.0    | 80677    | 80677    | non_maximum_suppression_kernel   |
| 17.7     | 79045      | 1         | 79045.0    | 79045    | 79045    | merge_gradients_kernel           |
| 5.4      | 24129      | 1         | 24129.0    | 24129    | 24129    | min_max_cuda                     |
| 2.5      | 11040      | 1         | 11040.0    | 11040    | 11040    | first_edges_kernel               |
| 1.9      | 8512       | 1         | 8512.0     | 8512     | 8512     | normalize_cuda                   |
| 0.9      | 3936       | 1         | 3936.0     | 3936     | 3936     | generate_gaussian_kernel         |

Overall, each kernel's runtime was what we expected, with convolution and hysteresis_edges taking the most time due to the number of iterations they had, while kernels like generate_gaussian_kernel or normalize being much faster due to their highly parallelizable nature.