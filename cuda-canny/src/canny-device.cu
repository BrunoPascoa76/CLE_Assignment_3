
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

__global__ void convolution_1d_rows(const pixel_t *in, pixel_t *out, const float *kernel, const int nx, const int ny, const int kn) {
    // get current coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int khalf = kn / 2;

    if (x >= nx || y >= ny)
        return; // out of bounds (if image size is not multiple of 16)
    
    float sum = 0.0f;

    for(int kx=-khalf; kx<=khalf; kx++) {
        int ix=x+kx;

        if(ix>=0 && ix<nx)
            sum += in[y * nx + ix] * kernel[khalf + kx];
    }
    out[y * nx + x] = (pixel_t)sum;
}

__global__ void convolution_1d_cols(const pixel_t *in, pixel_t *out, const float *kernel, const int nx, const int ny, const int kn) {
    // get current coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int khalf = kn / 2;

    if (x >= nx || y >= ny)
        return; // out of bounds (if image size is not multiple of 16)
    
    float sum = 0.0f;

     for (int ky = -khalf; ky <= khalf; ky++) {
        int iy = y + ky;

        if (iy >= 0 && iy < ny) {
            sum += in[iy * nx + x] * kernel[khalf + ky];
        }
    }
    out[y * nx + x] = (pixel_t)sum;
}

__global__ void min_max_kernel(const pixel_t *in, int* bmin, int* bmax, int totpixels) {
    extern __shared__ int shared[]; //for the merge
    int tid=threadIdx.x;
    int gid=blockIdx.x*blockDim.x+tid;

    //Load block into memory
    int val=(gid<totpixels)?in[gid]:INT_MAX;
    shared[tid]=val; //min
    shared[tid+blockDim.x]=val; //max
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1) { //trying to adapt the same idea as the mpi merge sort (just, without the sort) (basically every "iteration" one half compares its values with the other half's and then the number of participants is halved)
        if(tid<s) {
            shared[tid]=min(shared[tid],shared[tid+s]);
            shared[tid+blockDim.x]=max(shared[tid+blockDim.x],shared[tid+s+blockDim.x]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        bmin[blockIdx.x] = shared[0];
        bmax[blockIdx.x] = shared[blockDim.x];
    }
}

void min_max_cuda(const pixel_t *in, const int nx, const int ny, int* min_val, int* max_val) {
    int totpixels=nx*ny;

    dim3 dimBlock(256); //16*16
    dim3 gridSize((totpixels+dimBlock.x-1)/dimBlock.x);

    int *bmins, *bmaxs;
    cudaSafeCall(cudaMalloc(&bmins, gridSize.x*sizeof(int)));
    cudaSafeCall(cudaMalloc(&bmaxs, gridSize.x*sizeof(int)));

    int* h_bmins = (int*)malloc(gridSize.x * sizeof(int));
    int* h_bmaxs = (int*)malloc(gridSize.x * sizeof(int));

    min_max_kernel<<<gridSize, dimBlock, 2*dimBlock.x*sizeof(int)>>>(in, bmins, bmaxs, totpixels);
    cudaCheckMsg("min_max_kernel launch failed");

    cudaSafeCall(cudaMemcpy(h_bmins, bmins, gridSize.x*sizeof(int), cudaMemcpyDeviceToHost));
    cudaSafeCall(cudaMemcpy(h_bmaxs, bmaxs, gridSize.x*sizeof(int), cudaMemcpyDeviceToHost));

    *min_val = INT_MAX;
    *max_val = -INT_MAX;

    //just compare between blocks in the host itself
    for (int i = 0; i < gridSize.x; i++) {
        if (h_bmins[i] < *min_val)
            *min_val = h_bmins[i];
        if (h_bmaxs[i] > *max_val)
            *max_val = h_bmaxs[i];
    }

    cudaSafeCall(cudaFree(bmins));
    cudaSafeCall(cudaFree(bmaxs));
    free(h_bmins);
    free(h_bmaxs);
}

__global__ void normalize_kernel(pixel_t *inout, const int nx, const int ny, const int kn, const int min, const int max) {
    const int khalf = kn / 2;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < khalf || x >= nx - khalf || y < khalf || y >= ny - khalf)
        return; // out of bounds (if image size is not multiple of 16)

    inout[y*nx+x]=MAX_BRIGHTNESS * ((int)inout[y*nx+x] - (float)min) / ((float)max - (float)min);
}

void gaussian_filter_device(pixel_t *in,
                            const int nx, const int ny, const float sigma)
{
    const int n = 2 * (int)(2 * sigma) + 3;
    const float mean = (n - 1) / 2.0f;
    float kernel[n]; //in theory switching from 1 pass of a 2d kernel to 2 passes of a 1d kernel. In practive, however it did not work (perhaps it had poor cache utilization)
    float sum=0.0f;

    size_t c = 0;
    for(int i=0;i<n;i++){
        float x=i-mean;
        kernel[i]=expf(-0.5f * (x * x) / (sigma * sigma));
        sum+=kernel[i];
    }

    for (int i = 0; i < n; i++)
        kernel[i] /= sum;

    pixel_t* d_temp;
    cudaSafeCall(cudaMalloc(&d_temp, nx * ny * sizeof(pixel_t)));

    float* d_kernel;
    cudaSafeCall(cudaMalloc(&d_kernel, n * sizeof(float)));

    //copy over kernel
    cudaSafeCall(cudaMemcpy(d_kernel, kernel, n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(256,1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    convolution_1d_cols<<<grid, block>>>(in, d_temp, d_kernel, nx, ny, n);
    cudaCheckMsg("horizontal_convolution_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    block=dim3(1,256);
    grid=dim3(1,(ny + block.y - 1) / block.y);

    convolution_1d_rows<<<grid, block>>>(d_temp, in, d_kernel, nx, ny, n);
    cudaCheckMsg("vertical_convolution_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    int min, max;
    
    min_max_cuda(in, nx, ny, &min, &max);
    cudaCheckMsg("min_max_cuda launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    normalize_kernel<<<grid, block>>>(in, nx, ny, n, min, max);
    cudaCheckMsg("normalize_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    cudaSafeCall(cudaFree(d_temp));
    cudaSafeCall(cudaFree(d_kernel));
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

__global__ void merge_gradients_kernel(const pixel_t *after_Gx, const pixel_t *after_Gy, pixel_t *G, const int nx, const int ny)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || y < 1 || x >= nx - 1 || y >= ny - 1)
        return;

    int c = x + y * nx;
    G[c] = (pixel_t)(hypot((double)(after_Gx[c]), (double)(after_Gy[c])));
}

// canny edge detector code to run on the GPU
void cannyDevice(const int *h_idata, const int w, const int h,
                 const int tmin, const int tmax,
                 const float sigma,
                 int *h_odata)
{
    const int nx = w;
    const int ny = h;
    const int conv_kernel_size = 3;

    pixel_t *G = (pixel_t *)calloc(nx * ny, sizeof(pixel_t));
    pixel_t *after_Gx = (pixel_t *)calloc(nx * ny, sizeof(pixel_t));
    pixel_t *after_Gy = (pixel_t *)calloc(nx * ny, sizeof(pixel_t));
    pixel_t *nms = (pixel_t *)calloc(nx * ny, sizeof(pixel_t));

    pixel_t *input = NULL, *output = NULL, *d_Gx = NULL, *d_Gy = NULL, *d_nms = NULL, *d_G = NULL;
    float *kernel = NULL;

    cudaSafeCall(cudaMalloc(&input, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&output, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&kernel, sizeof(float) * conv_kernel_size * conv_kernel_size));
    cudaSafeCall(cudaMalloc(&d_G, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_Gx, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_Gy, sizeof(pixel_t) * nx * ny));
    cudaSafeCall(cudaMalloc(&d_nms, sizeof(pixel_t) * nx * ny));

    if (G == NULL || after_Gx == NULL || after_Gy == NULL ||
        nms == NULL || h_odata == NULL)
    {
        fprintf(stderr, "canny_edge_detection:"
                        " Failed memory allocation(s).\n");
        exit(1);
    }

    cudaSafeCall(cudaMemcpy(input, h_idata, nx * ny * sizeof(pixel_t), cudaMemcpyHostToDevice));

    // Gaussian filter
    gaussian_filter_device(input, nx, ny, sigma);

    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    // x gradient convolution

    const float Gx[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
    cudaSafeCall(cudaMemcpy(kernel, Gx, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    // call for x direction
    convolution_cuda_kernel<<<gridDim, blockDim>>>(input, d_Gx, kernel, nx, ny, conv_kernel_size);

    cudaCheckMsg("convolution_cuda_kernel X launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    const float Gy[] = {1, 2, 1,
                        0, 0, 0,
                        -1, -2, -1};
    cudaSafeCall(cudaMemcpy(kernel, Gy, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    convolution_cuda_kernel<<<gridDim, blockDim>>>(input, d_Gy, kernel, nx, ny, conv_kernel_size);
    cudaCheckMsg("convolution_cuda_kernel Y launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    merge_gradients_kernel<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, nx, ny);
    cudaCheckMsg("merge_gradients_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    non_maximum_suppression_kernel<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, d_nms, nx, ny);
    cudaCheckMsg("non_maximum_suppression_kernel launch failed");
    cudaSafeCall(cudaDeviceSynchronize());

    cudaSafeCall(cudaMemcpy(nms, d_nms, nx * ny * sizeof(pixel_t), cudaMemcpyDeviceToHost));


    // edges with nms >= tmax
    memset(h_odata, 0, sizeof(pixel_t) * nx * ny);
    first_edges(nms, h_odata, nx, ny, tmax);

    // edges with nms >= tmin && neighbor is edge
    bool changed;
    do
    {
        changed = false;
        hysteresis_edges(nms, h_odata, nx, ny, tmin, &changed);
    } while (changed == true);

    // Free device memory
    cudaSafeCall(cudaFree(input));
    cudaSafeCall(cudaFree(output));
    cudaSafeCall(cudaFree(kernel));
    cudaSafeCall(cudaFree(d_G));
    cudaSafeCall(cudaFree(d_Gx));
    cudaSafeCall(cudaFree(d_Gy));
    cudaSafeCall(cudaFree(d_nms));

    free(after_Gx);
    free(after_Gy);
    free(G);
    free(nms);
}
