
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// #include <stdio.h>
#include <iostream>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Generate a random float between min and max
//float randomFloat(float min, float max) {
//    float scale = static_cast<float>(rand()) / RAND_MAX;
//    return min + scale * (max - min);
//}
//
//// Draw the ants on the image
//Mat drawAnts(Mat image, float antPosition[][3], int numberOfAnts) {
//    for (int i = 0; i < numberOfAnts; i++) {
//        int x = antPosition[i][0];
//        int y = antPosition[i][1];
//
//        image.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
//    }
//    return image;
//}
//
//Mat fadeImage(Mat image) {
//	for (int i = 0; i < image.rows; i++) {
//		for (int j = 0; j < image.cols; j++) {
//			Vec3b pixel = image.at<Vec3b>(i, j);
//            if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
//				continue;
//			}
//			pixel[0] = pixel[0] * 0.99;
//			pixel[1] = pixel[1] * 0.99;
//			pixel[2] = pixel[2] * 0.99;
//			image.at<Vec3b>(i, j) = pixel;
//		}
//	}
//	return image;
//}
//
//// Update the position of the ants
//static void updateAntPosition(float antPosition[][3], int numberOfAnts) {
//	const float stepSize = 1;
//	const float angleChange = 0.1;
//
//	for (int i = 0; i < numberOfAnts; i++) {
//        antPosition[i][2] += randomFloat(-angleChange, angleChange);
//		antPosition[i][0] += stepSize * cos(antPosition[i][2]);
//        if (antPosition[i][0] < 0) antPosition[i][0] = 0;
//        if (antPosition[i][0] >= 1000) antPosition[i][0] = 999;
//
//		antPosition[i][1] += stepSize * sin(antPosition[i][2]);
//        if (antPosition[i][1] < 0) antPosition[i][1] = 0;
//		if (antPosition[i][1] >= 1000) antPosition[i][1] = 999;
//	}
//}



// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
