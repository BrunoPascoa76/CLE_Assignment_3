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

// CUDA kernel to generate Gaussian kernel on GPU
__global__ void generate_gaussian_kernel_3d(float *kernel, int n, float sigma)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;
    
    int i = idx / n;
    int j = idx % n;
    float mean = (float)floor(n / 2.0);
    
    kernel[idx] = expf(-0.5f * (powf((i - mean) / sigma, 2.0f) + 
                               powf((j - mean) / sigma, 2.0f))) / 
                  (2.0f * M_PI * sigma * sigma);
}

// 3D convolution kernel for multiple images
__global__ void convolution_cuda_kernel_3d(const pixel_t *in, pixel_t *out, const float *kernel, 
                                           const int nx, const int ny, const int nz, const int kn)
{
    // get current coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    int khalf = kn / 2;

    if(x < khalf || y < khalf || x >= nx - khalf || y >= ny - khalf || z >= nz)
        return;

    float sum = 0.0f;
    int image_size = nx * ny;
    int base_idx = z * image_size;

    // now do the actual convolution for this pixel
    for (int ky = -khalf; ky <= khalf; ky++)
    {
        for (int kx = -khalf; kx <= khalf; kx++)
        {
            int ix = x + kx;
            int iy = y + ky;

            if (ix >= 0 && ix < nx && iy >= 0 && iy < ny)
            {
                float val = in[base_idx + iy * nx + ix];
                float weight = kernel[(ky + khalf) * kn + (kx + khalf)];
                sum += val * weight;
            }
        }
    }

    out[base_idx + y * nx + x] = (pixel_t)sum;
}


// 3D min/max reduction kernel
__global__ void min_max_cuda_3d(const pixel_t *in, const int nx, const int ny, const int nz, 
                                pixel_t *min_vals, pixel_t *max_vals)
{
    extern __shared__ pixel_t sdata[];

    pixel_t *smin = sdata;
    pixel_t *smax = &sdata[blockDim.x * blockDim.y];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z;
    
    int image_size = nx * ny;
    int base_idx = z * image_size;

    // Initialize shared memory
    if(x >= nx || y >= ny) {
        smin[tid] = INT_MAX;
        smax[tid] = INT_MIN;
    } else {
        pixel_t val = in[base_idx + y * nx + x];
        smin[tid] = val;
        smax[tid] = val;
    }
    __syncthreads();
    
    // Parallel reduction
    for(int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if(tid < s) {
            smin[tid] = min(smin[tid], smin[tid + s]);
            smax[tid] = max(smax[tid], smax[tid + s]);
        }
        __syncthreads();
    }

    if(tid == 0) {
        atomicMin(&min_vals[z], smin[0]);
        atomicMax(&max_vals[z], smax[0]);
    }
}

// 3D normalization kernel
__global__ void normalize_cuda_3d(pixel_t *inout, const int nx, const int ny, const int nz, const int kn, 
                                  const pixel_t *min_vals, const pixel_t *max_vals)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    const int khalf = kn / 2;

    if(x < khalf || y < khalf || x >= nx - khalf || y >= ny - khalf || z >= nz)
        return;

    int image_size = nx * ny;
    int idx = z * image_size + y * nx + x;
    
    int min_val = min_vals[z];
    int max_val = max_vals[z];

    if (max_val != min_val) {
        pixel_t pixel = (pixel_t)(MAX_BRIGHTNESS * ((float)(inout[idx] - min_val) / (float)(max_val - min_val)));
        inout[idx] = pixel;
    } else {
        inout[idx] = 0;
    }
}

// 3D Gaussian filter
void gaussian_filter_cuda_3d(const pixel_t *in, pixel_t *out, const int nx, const int ny, const int nz, const float sigma)
{
    const int n = 2 * (int)(2 * sigma) + 3;
    
    fprintf(stderr, "gaussian_filter_3d: kernel size %d, sigma=%g, processing %d images\n", n, sigma, nz);

    // Allocate memory for kernel on GPU
    float *d_kernel;
    cudaSafeCall(cudaMalloc((void **)&d_kernel, n * n * sizeof(float)));

    // Generate Gaussian kernel on GPU
    int kernel_threads = 256;
    int kernel_blocks = (n * n + kernel_threads - 1) / kernel_threads;
    generate_gaussian_kernel_3d<<<kernel_blocks, kernel_threads>>>(d_kernel, n, sigma);
    cudaCheckMsg("generate_gaussian_kernel_3d launch failed");

    // Set up 3D grid and block dimensions for convolution
    dim3 block(16, 16, 1);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, nz);

    // Apply Gaussian filter using 3D convolution
    convolution_cuda_kernel_3d<<<grid, block>>>(in, out, d_kernel, nx, ny, nz, n);
    cudaCheckMsg("gaussian convolution 3d launch failed");

    // Find min and max values for normalization for each image
    pixel_t *d_max_vals, *d_min_vals;
    pixel_t *h_max_vals = (pixel_t*)malloc(nz * sizeof(pixel_t));
    pixel_t *h_min_vals = (pixel_t*)malloc(nz * sizeof(pixel_t));
    
    // Initialize host arrays
    for(int i = 0; i < nz; i++) {
        h_max_vals[i] = INT_MIN;
        h_min_vals[i] = INT_MAX;
    }

    cudaSafeCall(cudaMalloc(&d_max_vals, nz * sizeof(pixel_t)));
    cudaSafeCall(cudaMalloc(&d_min_vals, nz * sizeof(pixel_t)));

    cudaSafeCall(cudaMemcpy(d_max_vals, h_max_vals, nz * sizeof(pixel_t), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_min_vals, h_min_vals, nz * sizeof(pixel_t), cudaMemcpyHostToDevice));

    // Set up grid for min/max computation
    dim3 minmax_grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, nz);
    min_max_cuda_3d<<<minmax_grid, block, 2 * block.x * block.y * sizeof(pixel_t)>>>(out, nx, ny, nz, d_min_vals, d_max_vals);
    cudaCheckMsg("min_max_cuda_3d launch failed");

    // Copy min/max values back to host
    cudaSafeCall(cudaMemcpy(h_max_vals, d_max_vals, nz * sizeof(pixel_t), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_min_vals, d_min_vals, nz * sizeof(pixel_t), cudaMemcpyDeviceToHost));

    // Normalize the result for all images
    normalize_cuda_3d<<<grid, block>>>(out, nx, ny, nz, n, d_min_vals, d_max_vals);
    cudaCheckMsg("normalize_cuda_3d launch failed");

    // Clean up
    cudaSafeCall(cudaFree(d_kernel));
    cudaSafeCall(cudaFree(d_max_vals));
    cudaSafeCall(cudaFree(d_min_vals));
    free(h_max_vals);
    free(h_min_vals);
}

__global__ void non_maximum_suppression_kernel_3d(const pixel_t *after_Gx, const pixel_t *after_Gy, 
                                                  const pixel_t *G, pixel_t *nms, 
                                                  const int nx, const int ny, const int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1 || z >= nz)
        return;

    int image_size = nx * ny;
    int base_idx = z * image_size;
    int c = base_idx + x + y * nx;
    
    const int nn = c - nx;
    const int ss = c + nx;
    const int ww = c + 1;
    const int ee = c - 1;
    const int nw = nn + 1;
    const int ne = nn - 1;
    const int sw = ss + 1;
    const int se = ss - 1;

    const float dir = (float)(fmod(atan2f(after_Gy[c], after_Gx[c]) + M_PI, M_PI) / M_PI) * 8;

    if (((dir <= 1 || dir > 7) && G[c] > G[ee] && G[c] > G[ww]) || // 0 deg
        ((dir > 1 && dir <= 3) && G[c] > G[nw] && G[c] > G[se]) || // 45 deg
        ((dir > 3 && dir <= 5) && G[c] > G[nn] && G[c] > G[ss]) || // 90 deg
        ((dir > 5 && dir <= 7) && G[c] > G[ne] && G[c] > G[sw]))   // 135 deg
        nms[c] = G[c];
    else
        nms[c] = 0;
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


// 3D First edges kernel
__global__ void first_edges_kernel_3d(const pixel_t *nms, pixel_t *reference,
                                      const int nx, const int ny, const int nz, const int tmax)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1 || z >= nz)
        return;

    int image_size = nx * ny;
    size_t c = z * image_size + x + y * nx;
    
    if (nms[c] >= tmax)
    {
        reference[c] = MAX_BRIGHTNESS;
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

// 3D Hysteresis edges kernel
__global__ void hysteresis_edges_kernel_3d(const pixel_t *nms, pixel_t *reference,
                                           const int nx, const int ny, const int nz, 
                                           const int tmin, int *changed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1 || z >= nz)
        return;

    int image_size = nx * ny;
    size_t t = z * image_size + x + y * nx;

    if (nms[t] >= tmin && reference[t] == 0)
    {
        // Check all 8 neighboring pixels
        int nbs[8];
        nbs[0] = t - nx;     // nn
        nbs[1] = t + nx;     // ss
        nbs[2] = t + 1;      // ww
        nbs[3] = t - 1;      // ee
        nbs[4] = nbs[0] + 1; // nw
        nbs[5] = nbs[0] - 1; // ne
        nbs[6] = nbs[1] + 1; // sw
        nbs[7] = nbs[1] - 1; // se

        for (int k = 0; k < 8; k++)
        {
            if (reference[nbs[k]] != 0)
            {
                reference[t] = MAX_BRIGHTNESS;
                atomicExch(changed, 1);
                break;
            }
        }
    }
}

// 3D Merge gradients kernel
__global__ void merge_gradients_kernel_3d(const pixel_t *after_Gx, const pixel_t *after_Gy, 
                                          pixel_t *G, const int nx, const int ny, const int nz)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1 || z >= nz)
        return;

    int image_size = nx * ny;
    int c = z * image_size + x + y * nx;
    G[c] = (pixel_t)(hypot((double)(after_Gx[c]), (double)(after_Gy[c])));
}

void cannyDevice3D(const int **h_idata_array, const int w, const int h, const int num_images,
                   const int tmin, const int tmax, const float sigma,
                   int **h_odata_array)
{
    const int nx = w;
    const int ny = h;
    const int nz = num_images;
    const size_t image_size = nx * ny * sizeof(pixel_t);
    const size_t total_size = nx * ny * nz * sizeof(pixel_t);
    const int conv_kernel_size = 3;

    printf("Processing %d images of size %dx%d\n", nz, nx, ny);

    // Device memory pointers
    pixel_t *d_input = NULL, *d_temp = NULL, *d_Gx = NULL, *d_Gy = NULL;
    pixel_t *d_nms = NULL, *d_G = NULL, *d_reference = NULL;
    float *d_kernel = NULL;
    int *d_changed = NULL;

    // Allocate device memory
    cudaSafeCall(cudaMalloc(&d_input, total_size));
    cudaSafeCall(cudaMalloc(&d_temp, total_size));
    cudaSafeCall(cudaMalloc(&d_kernel, sizeof(float) * conv_kernel_size * conv_kernel_size));
    cudaSafeCall(cudaMalloc(&d_G, total_size));
    cudaSafeCall(cudaMalloc(&d_Gx, total_size));
    cudaSafeCall(cudaMalloc(&d_Gy, total_size));
    cudaSafeCall(cudaMalloc(&d_nms, total_size));
    cudaSafeCall(cudaMalloc(&d_reference, total_size));
    cudaSafeCall(cudaMalloc(&d_changed, sizeof(int)));

    // Copy all input images to device
    for(int i = 0; i < nz; i++) {
        cudaSafeCall(cudaMemcpy(d_input + i * nx * ny, h_idata_array[i], 
                               image_size, cudaMemcpyHostToDevice));
    }

    // Initialize reference buffer to zeros
    cudaSafeCall(cudaMemset(d_reference, 0, total_size));

    // Apply Gaussian filter (now fully on GPU for all images)
    gaussian_filter_cuda_3d(d_input, d_temp, nx, ny, nz, sigma);

    // Set up 3D grid and block dimensions
    dim3 blockDim(16, 16, 1);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, 
                 (ny + blockDim.y - 1) / blockDim.y, nz);

    // Compute gradients using Sobel operators
    const float Gx[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
    cudaSafeCall(cudaMemcpy(d_kernel, Gx, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    // X gradient for all images
    convolution_cuda_kernel_3d<<<gridDim, blockDim>>>(d_temp, d_Gx, d_kernel, nx, ny, nz, conv_kernel_size);
    cudaCheckMsg("convolution_cuda_kernel_3d X launch failed");

    const float Gy[] = {1, 2, 1,
                        0, 0, 0,
                        -1, -2, -1};
    cudaSafeCall(cudaMemcpy(d_kernel, Gy, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    // Y gradient for all images
    convolution_cuda_kernel_3d<<<gridDim, blockDim>>>(d_temp, d_Gy, d_kernel, nx, ny, nz, conv_kernel_size);
    cudaCheckMsg("convolution_cuda_kernel_3d Y launch failed");

    // Merge gradients to compute magnitude for all images
    merge_gradients_kernel_3d<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, nx, ny, nz);
    cudaCheckMsg("merge_gradients_kernel_3d launch failed");

    // Non-maximum suppression for all images
    non_maximum_suppression_kernel_3d<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, d_nms, nx, ny, nz);
    cudaCheckMsg("non_maximum_suppression_kernel_3d launch failed");

    // First edge detection (pixels >= tmax) for all images
    first_edges_kernel_3d<<<gridDim, blockDim>>>(d_nms, d_reference, nx, ny, nz, tmax);
    cudaCheckMsg("first_edges_kernel_3d launch failed");

    // Hysteresis edge linking for all images
    int h_changed;
    do {
        h_changed = 0;
        cudaSafeCall(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));
        
        hysteresis_edges_kernel_3d<<<gridDim, blockDim>>>(d_nms, d_reference, nx, ny, nz, tmin, d_changed);
        cudaCheckMsg("hysteresis_edges_kernel_3d launch failed");
        
        cudaSafeCall(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    } while (h_changed);

    // Copy results back to host
    for(int i = 0; i < nz; i++) {
        cudaSafeCall(cudaMemcpy(h_odata_array[i], d_reference + i * nx * ny, 
                               image_size, cudaMemcpyDeviceToHost));
    }

    // Free all device memory
    cudaSafeCall(cudaFree(d_input));
    cudaSafeCall(cudaFree(d_temp));
    cudaSafeCall(cudaFree(d_kernel));
    cudaSafeCall(cudaFree(d_G));
    cudaSafeCall(cudaFree(d_Gx));
    cudaSafeCall(cudaFree(d_Gy));
    cudaSafeCall(cudaFree(d_nms));
    cudaSafeCall(cudaFree(d_reference));
    cudaSafeCall(cudaFree(d_changed));
}