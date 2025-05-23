
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

    if(x < khalf || y < khalf || x >= nx - khalf || y >= ny - khalf)
        return; // the borders weren't touched in the cpu version, so trying to replicate that

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

__global__ void min_max_cuda(const pixel_t *in, const int nx, const int ny, pixel_t *min, pixel_t *max){
    extern __shared__ pixel_t sdata[];

    pixel_t *smin = sdata; //get the pointers for the min and max positions
    pixel_t *smax = &sdata[blockDim.x*blockDim.y];

    int tid=threadIdx.y*blockDim.x+threadIdx.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x >= nx || y >= ny)
        return;
        
    pixel_t val = in[y * nx + x];
    smin[tid]=val;
    smax[tid]=val;
    __syncthreads();
    

    for(int s=(blockDim.x * blockDim.y) / 2; s>0;s>>=1){ //parallel reduction (if doing atomicmin/max has too much contention, we simply reduce the number of values)
        if(tid < s){
            smin[tid]=min(smin[tid],smin[tid+s]);
            smax[tid]=max(smax[tid],smax[tid+s]);
        }
        __syncthreads();
    }

    if(tid == 0){ //now we only need to do atomicmin/max once per block (if it's still too much, I'll then do blockwise, but not for now)
        atomicMin(min, smin[0]);
        atomicMax(max, smax[0]);
    }
}

__global__ void normalize_cuda(pixel_t *inout, const int nx, const int ny, const int kn, const int min, const int max){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    const int khalf = kn / 2;

    if(x < khalf || y < khalf || x >= nx - khalf || y >= ny - khalf)
        return;

    pixel_t pixel= MAX_BRIGHTNESS * ((int)inout[y * nx + x] - (float)min) / ((float)max - (float)min);
    inout[y*nx+x]=pixel;
}

void gaussian_filter_cuda(const pixel_t *in, pixel_t *out,
                          const int nx, const int ny, const float sigma){ //this is still on host (it was just to visually separate the more complex gaussian from the host)
    const int n = 2 * (int)(2 * sigma) + 3;
    const float mean = (float)floor(n / 2.0);
    float kernel[n * n]; // variable length array

    fprintf(stderr, "gaussian_filter: kernel size %d, sigma=%g\n",
            n, sigma);
    size_t c = 0;
    for (int i = 0; i < n; i++) //we can still do this on cpu, as it's not worth the overhead
        for (int j = 0; j < n; j++)
        {
            kernel[c] = exp(-0.5 * (pow((i - mean) / sigma, 2.0) +
                                    pow((j - mean) / sigma, 2.0))) /
                        (2 * M_PI * sigma * sigma);
            c++;
        }

    float *d_kernel;

    cudaSafeCall(cudaMalloc((void **)&d_kernel, n * n * sizeof(float)));
    cudaSafeCall(cudaMemcpy(d_kernel, kernel, n * n * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    convolution_cuda_kernel<<<grid, block>>>(in, out, d_kernel, nx, ny, n);

    pixel_t *d_max, *d_min;
    pixel_t max=-INT_MAX, min=INT_MAX;

    cudaSafeCall(cudaMalloc(&d_max, sizeof(pixel_t)));
    cudaSafeCall(cudaMalloc(&d_min, sizeof(pixel_t)));

    cudaSafeCall(cudaMemcpy(d_max, &max, sizeof(pixel_t), cudaMemcpyHostToDevice));
    cudaSafeCall(cudaMemcpy(d_min, &min, sizeof(pixel_t), cudaMemcpyHostToDevice));

    min_max_cuda<<<grid,block,2*block.x*block.y*sizeof(pixel_t)>>>(out, nx, ny, d_min, d_max);

    int h_max, h_min;

    cudaSafeCall(cudaMemcpy(&h_max, d_max, sizeof(pixel_t), cudaMemcpyDeviceToHost)); //this is not temporary, it seems that copying to host and then passing them as constants is faster than using global memory, for some reason
    cudaSafeCall(cudaMemcpy(&h_min, d_min, sizeof(pixel_t), cudaMemcpyDeviceToHost));

    normalize_cuda<<<grid,block>>>(out, nx, ny, n, h_min, h_max);
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
    gaussian_filter_cuda(input,output, nx, ny, sigma);

    
    dim3 blockDim(16, 16);
    dim3 gridDim((nx + blockDim.x - 1) / blockDim.x, (ny + blockDim.y - 1) / blockDim.y);

    // x gradient convolution

    const float Gx[] = {-1, 0, 1,
                        -2, 0, 2,
                        -1, 0, 1};
    cudaSafeCall(cudaMemcpy(kernel, Gx, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    // call for x direction
    convolution_cuda_kernel<<<gridDim, blockDim>>>(output, d_Gx, kernel, nx, ny, conv_kernel_size);

    cudaCheckMsg("convolution_cuda_kernel X launch failed");
    //cudaSafeCall(cudaDeviceSynchronize()); (the synchronize do not seem to be needed, but uncomment if proven otherwise)

    const float Gy[] = {1, 2, 1,
                        0, 0, 0,
                        -1, -2, -1};
    cudaSafeCall(cudaMemcpy(kernel, Gy, conv_kernel_size * conv_kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    convolution_cuda_kernel<<<gridDim, blockDim>>>(output, d_Gy, kernel, nx, ny, conv_kernel_size);
    cudaCheckMsg("convolution_cuda_kernel Y launch failed");
    //cudaSafeCall(cudaDeviceSynchronize());

    merge_gradients_kernel<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, nx, ny);
    cudaCheckMsg("merge_gradients_kernel launch failed");
    //cudaSafeCall(cudaDeviceSynchronize());

    non_maximum_suppression_kernel<<<gridDim, blockDim>>>(d_Gx, d_Gy, d_G, d_nms, nx, ny);
    cudaCheckMsg("non_maximum_suppression_kernel launch failed");
    //cudaSafeCall(cudaDeviceSynchronize());


    //REMOVE: remove when first_edges is "cuda-fied"
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
