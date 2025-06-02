// Based on CUDA SDK template from NVIDIA
// Modified for 3D multi-image processing

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <assert.h>
#include <float.h>
#include <dirent.h>
#include <libgen.h>

//==============================
// Your code goes into this file
#include "canny-device.cu"
//==============================

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

//canny edge detector code to run on the host
void cannyHost( const int *h_idata, const int w, const int h,
               const int tmin,            // tmin canny parameter
               const int tmax,            // tmax canny parameter
               const float sigma,         // sigma canny parameter
               int * reference)
{
    const int nx = w;
    const int ny = h;

    pixel_t *G        = (pixel_t *) calloc(nx * ny, sizeof(pixel_t));
    pixel_t *after_Gx = (pixel_t *) calloc(nx * ny, sizeof(pixel_t));
    pixel_t *after_Gy = (pixel_t *) calloc(nx * ny, sizeof(pixel_t));
    pixel_t *nms      = (pixel_t *) calloc(nx * ny, sizeof(pixel_t));

    if (G == NULL || after_Gx == NULL || after_Gy == NULL ||
        nms == NULL || reference == NULL) {
        fprintf(stderr, "canny_edge_detection:"
                " Failed memory allocation(s).\n");
        exit(1);
    }

    // Gaussian filter
    gaussian_filter(h_idata, reference, nx, ny, sigma);

    const float Gx[] = {-1, 0, 1,
        -2, 0, 2,
        -1, 0, 1};

    // Gradient along x
    convolution(reference, after_Gx, Gx, nx, ny, 3);

    const float Gy[] = { 1, 2, 1,
        0, 0, 0,
        -1,-2,-1};

    // Gradient along y
    convolution(reference, after_Gy, Gy, nx, ny, 3);

    // Merging gradients
    for (int i = 1; i < nx - 1; i++)
        for (int j = 1; j < ny - 1; j++) {
            const int c = i + nx * j;
            G[c] = (pixel_t)(hypot((double)(after_Gx[c]), (double)( after_Gy[c]) ));
        }

    // Non-maximum suppression, straightforward implementation.
    non_maximum_supression(after_Gx, after_Gy, G, nms, nx, ny);

    // edges with nms >= tmax
    memset(reference, 0, sizeof(pixel_t) * nx * ny);
    first_edges(nms, reference, nx, ny, tmax);

    // edges with nms >= tmin && neighbor is edge
    bool changed;
    do {
        changed = false;
        hysteresis_edges(nms, reference, nx, ny, tmin, &changed);
    } while (changed==true);

    free(after_Gx);
    free(after_Gy);
    free(G);
    free(nms);
}

// Function to load multiple images from a directory or file list
int loadMultipleImages(char* input_path, int*** images, unsigned int* w, unsigned int* h, int max_images)
{
    struct dirent *entry;
    DIR *dp;
    char filepath[1024];
    int image_count = 0;
    
    // Check if input is a directory
    dp = opendir(input_path);
    if (dp != NULL) {
        printf("Processing directory: %s\n", input_path);
        
        // Allocate array for image pointers
        *images = (int**)malloc(max_images * sizeof(int*));
        
        while ((entry = readdir(dp)) && image_count < max_images) {
            // Skip . and .. entries
            if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
                continue;
            
            // Check for .pgm extension
            char *ext = strrchr(entry->d_name, '.');
            if (ext && strcmp(ext, ".pgm") == 0) {
                snprintf(filepath, sizeof(filepath), "%s/%s", input_path, entry->d_name);
                
                unsigned int img_w, img_h;
                if (loadPGM(filepath, (uint32_t**)&((*images)[image_count]), &img_w, &img_h) == 1) {
                    if (image_count == 0) {
                        *w = img_w;
                        *h = img_h;
                        printf("Image dimensions: %dx%d\n", *w, *h);
                    } else if (img_w != *w || img_h != *h) {
                        printf("Warning: Image %s has different dimensions (%dx%d), skipping\n",
                               entry->d_name, img_w, img_h);
                        free((*images)[image_count]);
                        continue;
                    }
                    printf("Loaded image %d: %s\n", image_count + 1, entry->d_name);
                    image_count++;
                } else {
                    printf("Failed to load image: %s\n", filepath);
                }
            }
        }
        closedir(dp);
    } else {
        // Treat as single file
        printf("Processing single file: %s\n", input_path);
        *images = (int**)malloc(1 * sizeof(int*));
        if (loadPGM(input_path, (uint32_t**)&((*images)[0]), w, h) == 1) {
            image_count = 1;
        } else {
            printf("Failed to load image: %s\n", input_path);
            free(*images);
            return 0;
        }
    }
    
    printf("Total images loaded: %d\n", image_count);
    return image_count;
}

// Function to save multiple images
void saveMultipleImages(char* output_prefix, int** images, unsigned int w, unsigned int h, int num_images, const char* suffix)
{
    char filename[1024];
    for (int i = 0; i < num_images; i++) {
        snprintf(filename, sizeof(filename), "%s_%s_%03d.pgm", output_prefix, suffix, i);
        if (savePGM(filename, (unsigned int*)images[i], w, h) != 1) {
            printf("Failed to save image: %s\n", filename);
        } else {
            printf("Saved: %s\n", filename);
        }
    }
}

// print command line format
void usage(char *command)
{
    printf("Usage: %s [-h] [-d device] [-i input_path] [-o output_prefix] [-n num_images] [-s sigma] [-t threshold]\n", command);
    printf("  -i input_path    : Input directory containing .pgm files or single .pgm file\n");
    printf("  -o output_prefix : Output filename prefix for processed images\n");
    printf("  -n num_images    : Maximum number of images to process (default: 10)\n");
    printf("  -s sigma         : Gaussian sigma parameter (default: 1.0)\n");
    printf("  -t tmin          : Lower threshold for hysteresis (default: 45)\n");
    printf("  -x tmax          : Upper threshold for hysteresis (default: 50)\n");
    printf("  -d device        : CUDA device ID (default: 0)\n");
    printf("  -h               : Show this help\n");
}

// main
int main( int argc, char** argv)
{
    // default command line options
    int deviceId = 0;
    char *input_path = (char *)"images/";
    char *output_prefix = (char *)"output";
    int tmin = 45, tmax = 50;
    float sigma = 1.0f;
    int max_images = 10;

    // parse command line arguments
    int opt;
    while( (opt = getopt(argc,argv,"d:i:o:n:x:t:s:h")) !=-1)
    {
        switch(opt)
        {
            case 'd':  // device
                if(sscanf(optarg,"%d",&deviceId)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;

            case 'i': // input path (directory or file)
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                input_path = strdup(optarg);
                break;
                
            case 'o': // output prefix
                if(strlen(optarg)==0)
                {
                    usage(argv[0]);
                    exit(1);
                }
                output_prefix = strdup(optarg);
                break;
                
            case 'n': // max number of images
                if(strlen(optarg)==0 || sscanf(optarg,"%d",&max_images)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
                
            case 't': // tmin
                if(strlen(optarg)==0 || sscanf(optarg,"%d",&tmin)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
                
            case 'x': // tmax
                if(strlen(optarg)==0 || sscanf(optarg,"%d",&tmax)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
                
            case 's': // sigma
                if(strlen(optarg)==0 || sscanf(optarg,"%f",&sigma)!=1)
                {
                    usage(argv[0]);
                    exit(1);
                }
                break;
                
            case 'h': // help
                usage(argv[0]);
                exit(0);
                break;
        }
    }

    // Banner
    int deviceCount = 0;
    cudaSafeCall( cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0){
        fprintf(stderr, "No CUDA-capable device found.");
        return 0;
    }

    printf("Available devices\n");
    for (int i = 0; i < deviceCount; ++i){
        cudaDeviceProp deviceProp;
        cudaSafeCall(cudaGetDeviceProperties(&deviceProp, i));

        if (i == deviceId) printf("->");
        else printf("  ");

        printf("%d: %s (compute %d.%d)\n", i, deviceProp.name, deviceProp.major, deviceProp.minor);
    }
    printf("\n");

    // select cuda device
    cudaSafeCall( cudaSetDevice( deviceId ) );

    // create events to measure host canny detector time and device canny detector time
    cudaEvent_t startH, stopH, startD, stopD;
    cudaEventCreate(&startH);
    cudaEventCreate(&stopH);
    cudaEventCreate(&startD);
    cudaEventCreate(&stopD);

    // Load multiple images
    int** h_idata_array = NULL;
    unsigned int w, h;
    int num_images = loadMultipleImages(input_path, &h_idata_array, &w, &h, max_images);
    
    if (num_images == 0) {
        printf("No images loaded. Exiting.\n");
        exit(1);
    }

    // allocate mem for the results on host side
    int** h_odata_array = (int**)malloc(num_images * sizeof(int*));
    int** reference_array = (int**)malloc(num_images * sizeof(int*));
    
    for (int i = 0; i < num_images; i++) {
        h_odata_array[i] = (int*)calloc(h * w, sizeof(int));
        reference_array[i] = (int*)calloc(h * w, sizeof(int));
    }

    // Process images on host (one by one for comparison)
    printf("\nProcessing on Host...\n");
    cudaEventRecord( startH, 0 );
    for (int i = 0; i < num_images; i++) {
        cannyHost(h_idata_array[i], w, h, tmin, tmax, sigma, reference_array[i]);
    }
    cudaEventRecord( stopH, 0 );
    cudaEventSynchronize( stopH );

    // Process all images on GPU simultaneously
    printf("Processing on Device (3D batch)...\n");
    cudaEventRecord( startD, 0 );
    cannyDevice3D((const int**)h_idata_array, w, h, num_images, tmin, tmax, sigma, h_odata_array);
    cudaEventRecord( stopD, 0 );
    cudaEventSynchronize( stopD );

    // check if kernel execution generated an error
    cudaCheckMsg("Kernel execution failed");

    float timeH, timeD;
    cudaEventElapsedTime( &timeH, startH, stopH );
    printf( "Host processing time: %f (ms) for %d images\n", timeH, num_images);
    printf( "Average host time per image: %f (ms)\n", timeH / num_images);
    
    cudaEventElapsedTime( &timeD, startD, stopD );
    printf( "Device processing time: %f (ms) for %d images\n", timeD, num_images);
    printf( "Average device time per image: %f (ms)\n", timeD / num_images);
    printf( "Speedup: %.2fx\n", timeH / timeD);

    // Save output images
    saveMultipleImages(output_prefix, reference_array, w, h, num_images, "host");
    saveMultipleImages(output_prefix, h_odata_array, w, h, num_images, "device");

    // Compare results
    int total_pixels = w * h * num_images;
    int total_diff = 0;
    
    for (int img = 0; img < num_images; img++) {
        int img_diff = 0;
        for (int i = 0; i < w * h; ++i){
            int delta = abs(reference_array[img][i] - h_odata_array[img][i]);
            if (delta != 0) {
                img_diff++;
                total_diff++;
            }
        }
        printf("Image %d: %d different pixels out of %d (%.2f%%)\n", 
               img, img_diff, w*h, img_diff/(float)(w*h) * 100.0f);
    }
    
    printf("\nOverall: %d different pixels out of %d (%.2f%%)\n", 
           total_diff, total_pixels, total_diff/(float)total_pixels * 100.0f);

    // cleanup memory
    for (int i = 0; i < num_images; i++) {
        free(h_idata_array[i]);
        free(h_odata_array[i]);
        free(reference_array[i]);
    }
    free(h_idata_array);
    free(h_odata_array);
    free(reference_array);

    cudaDeviceReset();
    
    return 0;
}
