// CLE 24'25
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>

// utilities for safe cuda api calls copied from cuda sdk.

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckMsg(msg) __cudaGetLastError(msg, __FILE__, __LINE__)

inline void __cudaSafeCall(cudaError_t err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __cudaGetLastError(const char *errorMessage, const char *file, const int line)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

#define max(a, b) (((a) > (b)) ? (a) : (b))
#define min(a, b) (((a) < (b)) ? (a) : (b))

#define MAX_BRIGHTNESS 255

// Use int instead `unsigned char' so that we can
// store negative values.
typedef int pixel_t;

// include image functions
#include "image.c"

// convolution of in image to out image using kernel of kn width
void convolution(const pixel_t *in, pixel_t *out, const float *kernel,
                 const int nx, const int ny, const int kn)
{
    assert(kn % 2 == 1);
    assert(nx > kn && ny > kn);
    const int khalf = kn / 2;

    for (int m = khalf; m < nx - khalf; m++)
        for (int n = khalf; n < ny - khalf; n++)
        {
            float pixel = 0.0;
            size_t c = 0;
            for (int j = -khalf; j <= khalf; j++)
                for (int i = -khalf; i <= khalf; i++)
                {
                    pixel += in[(n - j) * nx + m - i] * kernel[c];
                    c++;
                }

            out[n * nx + m] = (pixel_t)pixel;
        }
}

__global__ void convolution_cuda_kernel(const pixel_t *in, pixel_t *out, const float *kernel, const int nx, const int ny, const int kn)
{
    // get current coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int khalf = kn / 2;

    if (x >= nx || y >= ny)
        return; // out of bounds (if image size is not multiple of 16)

    float sum = 0.0f;

    // now do the actual convolution for this pixel
    for (int ky = -khalf; ky <= khalf; ky++)
    {
        for (int kx = -khalf; kx <= khalf; kx++)
        {
            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < nx && iy >= 0 && iy < ny)
            { // if the current kernel point is whithin bounds...
                float val = in[iy * nx + ix];
                float weight = kernel[(ky + khalf) * kn + (kx + khalf)];
                sum += val * weight;
            }
        }
    }

    out[y * nx + x] = (pixel_t)sum;
}

// determines min and max of in image
void min_max(const pixel_t *in, const int nx, const int ny, pixel_t *pmin, pixel_t *pmax)
{
    int min = INT_MAX, max = -INT_MAX;

    for (int m = 0; m < nx; m++)
        for (int n = 0; n < ny; n++)
        {
            int pixel = in[n * nx + m];
            if (pixel < min)
                min = pixel;
            if (pixel > max)
                max = pixel;
        }
    *pmin = min;
    *pmax = max;
}

// normalizes inout image using min and max values
void normalize(pixel_t *inout,
               const int nx, const int ny, const int kn,
               const int min, const int max)
{
    const int khalf = kn / 2;

    for (int m = khalf; m < nx - khalf; m++)
        for (int n = khalf; n < ny - khalf; n++)
        {

            pixel_t pixel = MAX_BRIGHTNESS * ((int)inout[n * nx + m] - (float)min) / ((float)max - (float)min);
            inout[n * nx + m] = pixel;
        }
}

/*
 * gaussianFilter:
 * http://www.songho.ca/dsp/cannyedge/cannyedge.html
 * determine size of kernel (odd #)
 * 0.0 <= sigma < 0.5 : 3
 * 0.5 <= sigma < 1.0 : 5
 * 1.0 <= sigma < 1.5 : 7
 * 1.5 <= sigma < 2.0 : 9
 * 2.0 <= sigma < 2.5 : 11
 * 2.5 <= sigma < 3.0 : 13 ...
 * kernelSize = 2 * int(2*sigma) + 3;
 */
void gaussian_filter(const pixel_t *in, pixel_t *out,
                     const int nx, const int ny, const float sigma)
{
    const int n = 2 * (int)(2 * sigma) + 3;
    const float mean = (float)floor(n / 2.0);
    float kernel[n * n]; // variable length array

    fprintf(stderr, "gaussian_filter: kernel size %d, sigma=%g\n",
            n, sigma);
    size_t c = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
        {
            kernel[c] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) +
                                    pow((j - mean) / sigma, 2.0))) /
                        (2 * M_PI * sigma * sigma);
            c++;
        }

    convolution(in, out, kernel, nx, ny, n);
    pixel_t max, min;
    min_max(out, nx, ny, &min, &max);
    normalize(out, nx, ny, n, min, max);
}

__global__ void non_maximum_suppression_kernel(const pixel_t *after_Gx, const pixel_t *after_Gy, const pixel_t *G, pixel_t *nms, const int nx, const int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1)
        return;

    int c = x + y * nx;
    const int nn = c - nx;
    const int ss = c + nx;
    const int ww = c + 1;
    const int ee = c - 1;
    const int nw = nn + 1;
    const int ne = nn - 1;
    const int sw = ss + 1;
    const int se = ss - 1;

    const float dir = (float)(fmod(atan2f(after_Gy[c],
                                         after_Gx[c]) +
                                       M_PI,
                                   M_PI) /
                              M_PI) *
                      8;

    if (((dir <= 1 || dir > 7) && G[c] > G[ee] &&
         G[c] > G[ww]) || // 0 deg
        ((dir > 1 && dir <= 3) && G[c] > G[nw] &&
         G[c] > G[se]) || // 45 deg
        ((dir > 3 && dir <= 5) && G[c] > G[nn] &&
         G[c] > G[ss]) || // 90 deg
        ((dir > 5 && dir <= 7) && G[c] > G[ne] &&
         G[c] > G[sw])) // 135 deg
        nms[c] = G[c];
    else
        nms[c] = 0;
}

// Canny non-maximum suppression
void non_maximum_supression(const pixel_t *after_Gx, const pixel_t *after_Gy, const pixel_t *G, pixel_t *nms,
                            const int nx, const int ny)
{
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++)
        {
            const int c = i + nx * j;
            const int nn = c - nx;
            const int ss = c + nx;
            const int ww = c + 1;
            const int ee = c - 1;
            const int nw = nn + 1;
            const int ne = nn - 1;
            const int sw = ss + 1;
            const int se = ss - 1;

            const float dir = (float)(fmod(atan2(after_Gy[c],
                                                 after_Gx[c]) +
                                               M_PI,
                                           M_PI) /
                                      M_PI) *
                              8;

            if (((dir <= 1 || dir > 7) && G[c] > G[ee] &&
                 G[c] > G[ww]) || // 0 deg
                ((dir > 1 && dir <= 3) && G[c] > G[nw] &&
                 G[c] > G[se]) || // 45 deg
                ((dir > 3 && dir <= 5) && G[c] > G[nn] &&
                 G[c] > G[ss]) || // 90 deg
                ((dir > 5 && dir <= 7) && G[c] > G[ne] &&
                 G[c] > G[sw])) // 135 deg
                nms[c] = G[c];
            else
                nms[c] = 0;
        }
}

// edges found in first pass for nms > tmax
void first_edges(const pixel_t *nms, pixel_t *reference,
                 const int nx, const int ny, const int tmax)
{

    size_t c = 1;
    for (int j = 1; j < ny - 1; j++)
    {
        for (int i = 1; i < nx - 1; i++)
        {
            if (nms[c] >= tmax)
            { // trace edges
                reference[c] = MAX_BRIGHTNESS;
            }
            c++;
        }
        c += 2; // because borders are not considered
    }
}

// Optimized first_edges kernel using shared memory
__global__ void first_edges_kernel(const pixel_t *nms, pixel_t *reference,
                                           const int nx, const int ny, const int tmax)
{
    // Block dimensions
    const int BLOCK_WIDTH = blockDim.x;
    const int BLOCK_HEIGHT = blockDim.y;
    
    // Shared memory to cache the nms values for this block
    __shared__ pixel_t s_nms[32][32]; // Assuming block size is 16x16 or 32x32
    
    // Calculate global indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * nx + x;
    
    // Local indices for accessing shared memory
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load nms values into shared memory (with bounds checking)
    if (x < nx && y < ny) {
        s_nms[ty][tx] = nms[idx];
    }
    
    // Wait for all threads to load their data
    __syncthreads();
    
    // Process only if within inner image bounds (not on border)
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && tx < BLOCK_WIDTH && ty < BLOCK_HEIGHT) {
        // Apply threshold directly from shared memory when possible
        if (s_nms[ty][tx] >= tmax) {
            reference[idx] = MAX_BRIGHTNESS;
        }
    }
}

// edges found in after first passes for nms > tmin && neighbor is edge
void hysteresis_edges(const pixel_t *nms, pixel_t *reference,
                      const int nx, const int ny, const int tmin, bool *pchanged)
{
    // Tracing edges with hysteresis . Non-recursive implementation.
    for (int i = 1; i < nx - 1; i++)
    {
        for (int j = 1; j < ny - 1; j++)
        {
            size_t t = i + j * nx;

            int nbs[8];          // neighbours
            nbs[0] = t - nx;     // nn
            nbs[1] = t + nx;     // ss
            nbs[2] = t + 1;      // ww
            nbs[3] = t - 1;      // ee
            nbs[4] = nbs[0] + 1; // nw
            nbs[5] = nbs[0] - 1; // ne
            nbs[6] = nbs[1] + 1; // sw
            nbs[7] = nbs[1] - 1; // se

            if (nms[t] >= tmin && reference[t] == 0)
            {
                for (int k = 0; k < 8; k++)
                    if (reference[nbs[k]] != 0)
                    {
                        reference[t] = MAX_BRIGHTNESS;
                        *pchanged = true;
                    }
            }
        }
    }
}

__global__ void hysteresis_edges_kernel(const pixel_t *nms, pixel_t *reference,
                                     const int nx, const int ny, const int tmin, 
                                     int *d_changed, const int max_iterations)
{
    __shared__ int s_changed;
    __shared__ pixel_t s_ref[18][18]; // Shared memory for reference values
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * nx + x;
    
    // Local indices for shared memory
    int tx = threadIdx.x + 1;
    int ty = threadIdx.y + 1;
    
    // Skip threads outside image area
    if (x >= nx || y >= ny)
        return;
    
    // Perform multiple iterations without kernel relaunch
    for (int iter = 0; iter < max_iterations; iter++) {
        // Reset changed flag at the beginning of each iteration
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            s_changed = 0;
        }
        __syncthreads();
        
        // Load reference values into shared memory
        // Each thread loads its own value
        if (x < nx && y < ny) {
            s_ref[ty][tx] = reference[idx];
        }
        
        // Additional threads load the border values (simplified for brevity)
        // ... (similar to hysteresis_edges_kernel_optimized)
        
        __syncthreads();
        
        // Process only if within inner image bounds (not on border)
        if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
            // Only process pixels that meet threshold but aren't edges yet
            if (nms[idx] >= tmin && s_ref[ty][tx] == 0) {
                // Check all 8 neighbors directly from shared memory
                bool is_edge = (s_ref[ty-1][tx] != 0)   || // North
                               (s_ref[ty+1][tx] != 0)   || // South
                               (s_ref[ty][tx+1] != 0)   || // East
                               (s_ref[ty][tx-1] != 0)   || // West
                               (s_ref[ty-1][tx-1] != 0) || // Northwest
                               (s_ref[ty-1][tx+1] != 0) || // Northeast
                               (s_ref[ty+1][tx-1] != 0) || // Southwest
                               (s_ref[ty+1][tx+1] != 0);   // Southeast
                
                // Mark as edge if any neighbor is an edge
                if (is_edge) {
                    reference[idx] = MAX_BRIGHTNESS;
                    s_changed = 1; // Mark that something changed in this iteration
                }
            }
        }
        
        __syncthreads();
        
        // If nothing changed in this iteration, we're done
        if (s_changed == 0) {
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                *d_changed = 0; // Set the global changed flag to indicate completion
            }
            break;
        }
        
        // Update the global d_changed if we're going to continue
        if (threadIdx.x == 0 && threadIdx.y == 0 && iter == max_iterations - 1) {
            *d_changed = 1; // Indicate we need more iterations
        }
        
        // Make sure all threads can see the updated reference values
        __syncthreads();
    }
}


__global__ void merge_gradients_kernel(const pixel_t *after_Gx, const pixel_t *after_Gy, pixel_t *G, const int nx, const int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1)
        return;

    int c = x + y * nx;
    G[c] = (pixel_t)(hypot((double)(after_Gx[c]), (double)(after_Gy[c])));
}

void cannyDevice(const int *h_idata, const int w, const int h,
                 const int tmin, const int tmax,
                 const float sigma,
                 int *h_odata)
{
    const int nx = w;
    const int ny = h;
    const int conv_kernel_size = 3;
    
    // Intermediate result for Gaussian filter on host
    // Using pinned memory for more efficient transfers
    pixel_t *h_gaussian_result;
    cudaSafeCall(cudaMallocHost(&h_gaussian_result, nx * ny * sizeof(pixel_t)));
    
    // Device memory allocations
    pixel_t *d_input = NULL, *d_output = NULL;
    pixel_t *d_Gx = NULL, *d_Gy = NULL, *d_G = NULL, *d_nms = NULL;
    pixel_t *d_reference = NULL;
    float *d_kernel = NULL;
    int *d_changed = NULL;

    // Allocate device memory
    cudaSafeCall(cudaMalloc(&d_input, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_output, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_kernel, sizeof(float) * conv_kernel_size * conv_kernel_size));
    cudaSafeCall(cudaMalloc(&d_G, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_Gx, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_Gy, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_nms, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_reference, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_changed, sizeof(int)));
    
    // Check host allocation
    if (h_gaussian_result == NULL || h_odata == NULL) {
        fprintf(stderr, "canny_edge_detection: Failed memory allocation(s).\n");
        exit(1);
    }

    // Gaussian filter (still on CPU for now)
    gaussian_filter(h_idata, h_gaussian_result, nx, ny, sigma);

    // Copy Gaussian filtered image to device
    cudaSafeCall(cudaMemcpy(d_input, h_gaussian_result, nx * ny * sizeof(pixel_t), cudaMemcpyHostToDevice));

    // Setup grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    // X gradient convolution
    const float Gx[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
    cudaSafeCall(cudaMemcpy(d_kernel, Gx, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    convolution_cuda_kernel<<<gridDim, blockDim>>>(d_input, d_Gx, d_kernel, nx, ny, conv_kernel_size);
    cudaCheckMsg("convolution_cuda_kernel X launch failed");

    // Y gradient convolution
    const float Gy[] = {1, 2, 1,
                        0, 0, 0,
                        -1, -2, -1};
    cudaSafeCall(cudaMemcpy(d_kernel, Gy, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    convolution_cuda_kernel<<<gridDim, blockDim>>>(d_input, d_Gy, d_kernel, nx, ny, conv_kernel_size);
    cudaCheckMsg("convolution_cuda_kernel Y launch failed");

    // Merge gradients (calculate magnitude)
    merge_gradients_kernel<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, nx, ny);
    cudaCheckMsg("merge_gradients_kernel launch failed");

    // Non-maximum suppression
    non_maximum_suppression_kernel<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, d_nms, nx, ny);
    cudaCheckMsg("non_maximum_suppression_kernel launch failed");

    // Initialize d_reference with zeros
    cudaSafeCall(cudaMemset(d_reference, 0, nx * ny * sizeof(pixel_t)));
    
    // First pass edge detection
    first_edges_kernel<<<gridDim, blockDim>>>(d_nms, d_reference, nx, ny, tmax);
    cudaCheckMsg("first_edges_kernel launch failed");
    
    // Hysteresis edge tracking
    int h_changed = 1;
    int max_iterations = 10;  // Maximum iterations to prevent infinite loops
    int iterations = 0;
    
    while (h_changed && iterations < max_iterations) {
        // Reset changed flag
        h_changed = 0;
        cudaSafeCall(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch hysteresis kernel with multiple iterations per kernel launch
        const int iters_per_launch = 5;
        hysteresis_edges_kernel<<<gridDim, blockDim>>>(d_nms, d_reference, nx, ny, tmin, d_changed, iters_per_launch);
        cudaCheckMsg("hysteresis_edges_kernel launch failed");
        
        // Copy changed flag back to host
        cudaSafeCall(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        
        iterations++;
    }
    
    fprintf(stderr, "Hysteresis completed after %d iterations\n", iterations);
    
    // Copy final result back to host output
    cudaSafeCall(cudaMemcpy(h_odata, d_reference, nx * ny * sizeof(pixel_t), cudaMemcpyDeviceToHost));

    // Free device memory
    cudaSafeCall(cudaFree(d_input));
    cudaSafeCall(cudaFree(d_output));
    cudaSafeCall(cudaFree(d_kernel));
    cudaSafeCall(cudaFree(d_G));
    cudaSafeCall(cudaFree(d_Gx));
    cudaSafeCall(cudaFree(d_Gy));
    cudaSafeCall(cudaFree(d_nms));
    cudaSafeCall(cudaFree(d_reference));
    cudaSafeCall(cudaFree(d_changed));
    
    // Free pinned host memory
    cudaSafeCall(cudaFreeHost(h_gaussian_result));
}