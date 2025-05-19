
    // Based on CUDA SDK template from NVIDIA

    // includes, system
    #include <stdlib.h>
    #include <stdio.h>
    #include <string.h>
    #include <math.h>
    #include <unistd.h>
    #include <assert.h>
    #include <float.h>


    //==============================
    // Your code goes into this file
    #include "canny-device.cu"
    //==============================

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

    // print command line format
    void usage(char *command)
    {
        printf("Usage: %s [-h] [-d device] [-i inputfile] [-o outputfile] [-r referenceFile] [-s sigma] [-t threshold]\n",command);
    }

    // main
    int main( int argc, char** argv)
    {
        // default command line options
        int deviceId = 0;
        char *fileIn=(char *)"images/lake.pgm",*fileOut=(char *)"out.pgm",*referenceOut=(char *)"reference.pgm";
        int tmin = 45, tmax = 50;
        float sigma=1.0f;

        // parse command line arguments
        int opt;
        while( (opt = getopt(argc,argv,"d:i:o:r:n:x:s:h")) !=-1)
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

                case 'i': // input image filename
                    if(strlen(optarg)==0)
                    {
                        usage(argv[0]);
                        exit(1);
                    }

                    fileIn = strdup(optarg);
                    break;
                case 'o': // output image (from device) filename
                    if(strlen(optarg)==0)
                    {
                        usage(argv[0]);
                        exit(1);
                    }
                    fileOut = strdup(optarg);
                    break;
                case 'r': // output image (from host) filename
                    if(strlen(optarg)==0)
                    {
                        usage(argv[0]);
                        exit(1);
                    }
                    referenceOut = strdup(optarg);
                    break;
                case 'n': // tmin
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

        // allocate host memory
        int* h_idata=NULL;
        unsigned int h,w;

        //load pgm
        if (loadPGM(fileIn, (uint32_t**)&h_idata, &w, &h) != 1) {
            exit(1);
        }

        // allocate mem for the result on host side
        int* h_odata = (int*) calloc( h*w, sizeof(unsigned int));
        int* reference = (int*) calloc( h*w, sizeof(unsigned int));

        // detect edges at host
        cudaEventRecord( startH, 0 );
        cannyHost(h_idata, w, h, tmin, tmax, sigma, reference);
        cudaEventRecord( stopH, 0 );
        cudaEventSynchronize( stopH );

        // detect edges at GPU
        cudaEventRecord( startD, 0 );
        cannyDevice(h_idata, w, h, tmin, tmax, sigma, h_odata);
        cudaEventRecord( stopD, 0 );
        cudaEventSynchronize( stopD );

        // check if kernel execution generated and error
        cudaCheckMsg("Kernel execution failed");

        float timeH, timeD;
        cudaEventElapsedTime( &timeH, startH, stopH );
        printf( "Host processing time: %f (ms)\n", timeH);
        cudaEventElapsedTime( &timeD, startD, stopD );
        printf( "Device processing time: %f (ms)\n", timeD);

        // save output images
        if (savePGM(referenceOut, (unsigned int *)reference, w, h) != 1) {
            exit(1);
        }

        if (savePGM(fileOut,(unsigned int *) h_odata, w, h) != 1) {
            exit(1);
        }

        int hist = 0;
        for (int i = 0; i < w * h; ++i){
            int delta = abs(reference[i] - h_odata[i]);
            if (delta != 0)
                hist++;
        }
        printf("\nNumber of different pixels: %d/%d (%.2f%%)\n", hist, w*h, hist/(float)(w*h) * 100.0f);

        // cleanup memory
        if (h_idata != nullptr)
            free( h_idata);

        free( h_odata);
        free( reference);

        cudaDeviceReset();
    }