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

// GPU implementation of min_max function
__global__ void min_max_kernel(const pixel_t *in, const int nx, const int ny, 
                              int *min_values, int *max_values, const int block_size) {
    extern __shared__ int shared_memory[];
    int *shared_min = shared_memory;
    int *shared_max = shared_memory + blockDim.x * blockDim.y;
    
    // Calculate thread position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Initialize with extreme values
    shared_min[tid] = INT_MAX;
    shared_max[tid] = -INT_MAX;
    
    // Process pixels within bounds
    if (x < nx && y < ny) {
        int pixel = in[y * nx + x];
        shared_min[tid] = pixel;
        shared_max[tid] = pixel;
    }
    
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    // Write results to global memory
    if (tid == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        min_values[block_id] = shared_min[0];
        max_values[block_id] = shared_max[0];
    }
}

// Helper function to find global min/max from block-level results
__global__ void reduce_min_max(int *min_values, int *max_values, int count) {
    extern __shared__ int shared_memory2[];
    int *shared_min = shared_memory2;
    int *shared_max = shared_memory2 + blockDim.x;
    
    int tid = threadIdx.x;
    
    // Initialize shared memory
    shared_min[tid] = (tid < count) ? min_values[tid] : INT_MAX;
    shared_max[tid] = (tid < count) ? max_values[tid] : -INT_MAX;
    
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < count) {
            shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    // Write final result
    if (tid == 0) {
        min_values[0] = shared_min[0];
        max_values[0] = shared_max[0];
    }
}

// Main function for finding min/max on GPU
void min_max_cuda(const pixel_t *d_in, const int nx, const int ny, pixel_t *pmin, pixel_t *pmax) {
    dim3 block_size(16, 16);
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, 
                  (ny + block_size.y - 1) / block_size.y);
    
    int num_blocks = grid_size.x * grid_size.y;
    
    // Allocate memory for block-level results
    int *d_min_values, *d_max_values;
    cudaSafeCall(cudaMalloc(&d_min_values, num_blocks * sizeof(int)));
    cudaSafeCall(cudaMalloc(&d_max_values, num_blocks * sizeof(int)));
    
    // Shared memory size for block-level reduction
    int shared_mem_size = 2 * block_size.x * block_size.y * sizeof(int);
    
    // Find min/max for each block
    min_max_kernel<<<grid_size, block_size, shared_mem_size>>>(
        d_in, nx, ny, d_min_values, d_max_values, block_size.x * block_size.y);
    cudaCheckMsg("min_max_kernel launch failed");
    
    // If we have multiple blocks, reduce to a single min/max pair
    if (num_blocks > 1) {
        // Choose an appropriate block size for the reduction
        int reduction_block_size = 256;
        int reduction_shared_mem = 2 * reduction_block_size * sizeof(int);
        
        reduce_min_max<<<1, reduction_block_size, reduction_shared_mem>>>(
            d_min_values, d_max_values, num_blocks);
        cudaCheckMsg("reduce_min_max launch failed");
    }
    
    // Copy results back to host
    int h_min, h_max;
    cudaSafeCall(cudaMemcpy(&h_min, d_min_values, sizeof(int), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(&h_max, d_max_values, sizeof(int), cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaSafeCall(cudaFree(d_min_values));
    cudaSafeCall(cudaFree(d_max_values));
    
    // Set output values
    *pmin = h_min;
    *pmax = h_max;
}

// GPU implementation of normalize function
__global__ void normalize_kernel(pixel_t *inout, const int nx, const int ny, 
                                const int kn, const int min, const int max) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int khalf = kn / 2;
    
    // Check if we're within bounds (and not in the border region that should be skipped)
    if (x >= khalf && x < nx - khalf && y >= khalf && y < ny - khalf) {
        int idx = y * nx + x;
        // Same normalization logic as the CPU version
        float norm_factor = (float)(MAX_BRIGHTNESS) / ((float)max - (float)min);
        pixel_t pixel = norm_factor * ((int)inout[idx] - (float)min);
        inout[idx] = pixel;
    }
}

// Main function for normalizing on GPU
void normalize_cuda(pixel_t *d_inout, const int nx, const int ny, 
                   const int kn, const int min, const int max) {
    dim3 block_size(16, 16);
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, 
                  (ny + block_size.y - 1) / block_size.y);
    
    normalize_kernel<<<grid_size, block_size>>>(d_inout, nx, ny, kn, min, max);
    cudaCheckMsg("normalize_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());
}

// Helper function to create Gaussian kernel on GPU
__global__ void create_gaussian_kernel(float *kernel, int n, float sigma) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n * n) {
        int i = idx / n;
        int j = idx % n;
        float mean = floorf(n / 2.0f);
        
        kernel[idx] = expf(-0.5f * (powf((i - mean) / sigma, 2.0f) + 
                                    powf((j - mean) / sigma, 2.0f))) / 
                     (2.0f * M_PI * sigma * sigma);
    }
}

// Complete GPU implementation of gaussian_filter
void gaussian_filter_cuda(const pixel_t *h_in, pixel_t *h_out,
                         const int nx, const int ny, const float sigma) {
    // Determine kernel size based on sigma
    const int n = 2 * (int)(2 * sigma) + 3;

    fprintf(stderr, "gaussian_filter_cuda: kernel size %d, sigma=%g\n", n, sigma);
    
    // Allocate device memory
    pixel_t *d_in, *d_out;
    float *d_kernel;
    
    cudaSafeCall(cudaMalloc(&d_in, nx * ny * sizeof(pixel_t)));
    cudaSafeCall(cudaMalloc(&d_out, nx * ny * sizeof(pixel_t)));
    cudaSafeCall(cudaMalloc(&d_kernel, n * n * sizeof(float)));
    
    // Copy input image to device
    cudaSafeCall(cudaMemcpy(d_in, h_in, nx * ny * sizeof(pixel_t), cudaMemcpyHostToDevice));
    
    // Create Gaussian kernel on GPU
    int threads_per_block = 256;
    int num_blocks = (n * n + threads_per_block - 1) / threads_per_block;
    
    create_gaussian_kernel<<<num_blocks, threads_per_block>>>(d_kernel, n, sigma);
    cudaCheckMsg("create_gaussian_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());
    
    // Set up grid and block dimensions for the convolution
    dim3 block_size(16, 16);
    dim3 grid_size((nx + block_size.x - 1) / block_size.x, 
                  (ny + block_size.y - 1) / block_size.y);
    
    // Perform convolution on GPU
    convolution_cuda_kernel<<<grid_size, block_size>>>(d_in, d_out, d_kernel, nx, ny, n);
    cudaCheckMsg("convolution_cuda_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());
    
    // Find min and max values
    pixel_t min_val, max_val;
    min_max_cuda(d_out, nx, ny, &min_val, &max_val);
    
    // Normalize the output
    normalize_cuda(d_out, nx, ny, n, min_val, max_val);
    
    // Copy result back to host
    cudaSafeCall(cudaMemcpy(h_out, d_out, nx * ny * sizeof(pixel_t), cudaMemcpyDeviceToHost));
    
    // Free device memory
    cudaSafeCall(cudaFree(d_in));
    cudaSafeCall(cudaFree(d_out));
    cudaSafeCall(cudaFree(d_kernel));
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

__global__ void first_edges_kernel(const pixel_t *nms, pixel_t *reference,
                                  const int nx, const int ny, const int tmax)
{
    // Calculate the global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within image bounds and not on the border
    if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1) {
        int idx = y * nx + x;
        
        // Same logic as CPU version: if nms >= tmax, mark as edge
        if (nms[idx] >= tmax) {
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
                              const int nx, const int ny, const int tmin, int *d_changed)
{
    // Calculate thread position
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within image bounds and not on border
    if (x <= 0 || y <= 0 || x >= nx-1 || y >= ny-1)
        return;
        
    int t = x + y * nx;
    
    // Only process pixels that meet the threshold but aren't yet marked as edges
    if (nms[t] >= tmin && reference[t] == 0)
    {
        // Check all 8 neighbors
        int nbs[8];          // neighbours
        nbs[0] = t - nx;     // nn
        nbs[1] = t + nx;     // ss
        nbs[2] = t + 1;      // ww
        nbs[3] = t - 1;      // ee
        nbs[4] = nbs[0] + 1; // nw
        nbs[5] = nbs[0] - 1; // ne
        nbs[6] = nbs[1] + 1; // sw
        nbs[7] = nbs[1] - 1; // se
        
        // Check if any neighbor is an edge
        for (int k = 0; k < 8; k++)
        {
            if (reference[nbs[k]] != 0)
            {
                reference[t] = MAX_BRIGHTNESS;
                *d_changed = 1; // Set the changed flag
                break;
            }
        }
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

    // Allocate device memory
    pixel_t *d_input = NULL, *d_output = NULL, *d_Gx = NULL, *d_Gy = NULL;
    pixel_t *d_nms = NULL, *d_G = NULL, *d_reference = NULL;
    float *d_kernel = NULL;
    int *d_changed = NULL;

    cudaSafeCall(cudaMalloc(&d_input, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_output, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_kernel, sizeof(float) * conv_kernel_size * conv_kernel_size));
    cudaSafeCall(cudaMalloc(&d_G, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_Gx, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_Gy, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_nms, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_reference, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_changed, sizeof(int)));

    // Copy input data to device
    cudaSafeCall(cudaMemcpy(d_input, h_idata, nx * ny * sizeof(pixel_t), cudaMemcpyHostToDevice));

    // Apply Gaussian filter (completely on GPU)
    gaussian_filter_cuda(h_idata, h_odata, nx, ny, sigma);
    
    // Copy filtered image back to device for further processing
    cudaSafeCall(cudaMemcpy(d_input, h_odata, nx * ny * sizeof(pixel_t), cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    // x gradient convolution
    const float Gx[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
    cudaSafeCall(cudaMemcpy(d_kernel, Gx, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    // call for x direction
    convolution_cuda_kernel<<<gridDim, blockDim>>>(d_input, d_Gx, d_kernel, nx, ny, conv_kernel_size);
    cudaCheckMsg("convolution_cuda_kernel X launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    // y gradient convolution
    const float Gy[] = {1, 2, 1,
                        0, 0, 0,
                        -1, -2, -1};
    cudaSafeCall(cudaMemcpy(d_kernel, Gy, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    convolution_cuda_kernel<<<gridDim, blockDim>>>(d_input, d_Gy, d_kernel, nx, ny, conv_kernel_size);
    cudaCheckMsg("convolution_cuda_kernel Y launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    // Calculate gradient magnitude
    merge_gradients_kernel<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, nx, ny);
    cudaCheckMsg("merge_gradients_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    // Non-maximum suppression
    non_maximum_suppression_kernel<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, d_nms, nx, ny);
    cudaCheckMsg("non_maximum_suppression_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    // Initialize d_reference with zeros
    cudaSafeCall(cudaMemset(d_reference, 0, nx * ny * sizeof(pixel_t)));
    
    // First pass of edge detection
    first_edges_kernel<<<gridDim, blockDim>>>(d_nms, d_reference, nx, ny, tmax);
    cudaCheckMsg("first_edges_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());
    
    // Hysteresis edge tracking
    int h_changed = 1;
    int iterations = 0;
    
    while (h_changed) {
        // Reset changed flag
        h_changed = 0;
        cudaSafeCall(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        
        // Launch hysteresis kernel
        hysteresis_edges_kernel<<<gridDim, blockDim>>>(d_nms, d_reference, nx, ny, tmin, d_changed);
        cudaCheckMsg("hysteresis_edges_kernel launch failed");
        cudaSafeCall(cudaDeviceSynchronize());
        
        // Copy changed flag back to host
        cudaSafeCall(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        
        iterations++;
    }
    
    fprintf(stderr, "Hysteresis completed after %d iterations\n", iterations);
    
    // Copy the final result back to host
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
}