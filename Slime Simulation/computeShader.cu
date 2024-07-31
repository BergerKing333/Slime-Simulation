#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <curand_kernel.h>
#include <iostream>

#include "computeShader.cuh"


using namespace cv;
using namespace std;

__device__ float sense(float* antPosition, float angleOffset, uchar1* image, int width, int height) {
    float sensorAngle = antPosition[2] + angleOffset;
    float2 sensorDir = make_float2(cos(sensorAngle), sin(sensorAngle));
    int2 sensorCenter = make_int2(antPosition[0] + sensorDir.x * sensorOffsetDst, antPosition[1] + sensorDir.y * sensorOffsetDst);
    float sum = 0;
    for (int offsetX = -sensorSize; offsetX <= sensorSize; ++offsetX) {
		for (int offsetY = -sensorSize; offsetY <= sensorSize; ++offsetY) {
			int2 pos = make_int2(sensorCenter.x + offsetX, sensorCenter.y + offsetY);

            if (pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height) continue;
            uchar1 pixel = image[pos.y * width + pos.x];
            sum += pixel.x;
		}
	}

    return sum;
}

__device__ float randomFloatKernel(curandState* state, float min, float max) {
    float scale = curand_uniform(state);
    return min + scale * (max - min);
}

__device__ uchar1 lerp(uchar1 a, uchar1 b, float t) {
    uchar1 result;
    result.x = static_cast<unsigned char>(a.x + t * (b.x - a.x));
    return result;
}

__global__ void fadeImageKernel(uchar1* image, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= rows || y >= cols) return;

    // Calculate fade factor (constant for all pixels)
    

    // Apply fade to pixel
    uchar1& pixel = image[y * rows + x];
    if (pixel.x == 0) return;
    pixel.x = static_cast<unsigned char>(pixel.x * fadeFactor);
}

__global__ void diffuseKernel(uchar1* image, uchar1* imageOutput, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < 0 || idx >= width || idy < 0 || idy >= height) return;

    float1 sum = make_float1(0);
    float1 originalPixel = make_float1(image[idy * width + idx].x);

    for (int offsetX = -1; offsetX <= 1; ++offsetX) {
        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            int x = min(width - 1, max(0, idx + offsetX));
            int y = min(height - 1, max(0, idy + offsetY));
            uchar1 pixel = image[y * width + x];
            sum.x += pixel.x;
        }
    }
    sum.x /= 9;
    float1 blurredCol = make_float1(originalPixel.x * (1 - diffuseRate) + sum.x * diffuseRate);
    blurredCol.x = max(0.0f, blurredCol.x - decayRate);
    
    imageOutput[idy * width + idx] = make_uchar1(blurredCol.x);
}

__global__ void blurImageKernel(uchar1* imageInput, uchar1* imageOutput, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < 0 || idx >= width || idy < 0 || idy >= height) return;

    uchar1 originalPixel = imageInput[idy * width + idx];
    uchar1 sum = make_uchar1(0);

    for (int offsetX = -1; offsetX <= 1; ++offsetX) {
		for (int offsetY = -1; offsetY <= 1; ++offsetY) {
			int x = min(width - 1, max(0, idx + offsetX));
			int y = min(height - 1, max(0, idy + offsetY));
			uchar1 pixel = imageInput[y * width + x];
			sum.x += pixel.x;
		}
	}
    sum.x /= 9;
    // uchar1 diffused_value = sum;
    uchar1 diffused_value = lerp(originalPixel, sum, lerpRate);
    imageOutput[idy * width + idx] = make_uchar1(diffused_value.x);
}

__global__ void drawAntsKernel(uchar1* image, int width, int height, float* antPositions, int numAnts) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAnts) return;
    int x = static_cast<int>(antPositions[idx * 3]);
    int y = static_cast<int>(antPositions[idx * 3 + 1]);
    image[y * width + x] = make_uchar1(255);
}

__global__ void updateAntPositionKernel(curandState* state, float* antPosition, int numAnts, int width, int height, uchar1* image) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAnts) return;

    float ant[3] = { antPosition[idx * 3], antPosition[idx * 3 + 1], antPosition[idx * 3 + 2] };
    float angle = ant[2];

    float weightForward = sense(ant, 0, image, width, height);
    float weightLeft = sense(ant, .1, image, width, height);
    float weightRight = sense(ant, -.1, image, width, height);

    float randomSteerStrength = randomFloatKernel(state, 0, 1);

    if (weightForward > weightLeft && weightForward > weightRight) {
        angle += 0;
	} else if (weightForward < weightRight && weightForward < weightLeft) {
        angle += (randomSteerStrength - 0.5) * 2 * turnSpeed;
	} else if (weightRight > weightLeft) {
        angle -= randomSteerStrength * turnSpeed;
    } else {
        angle += randomSteerStrength * turnSpeed;
    }

    antPosition[idx * 3 + 2] = angle;

    antPosition[idx * 3] += cos(angle) * speed;
    antPosition[idx * 3 + 1] += sin(angle) * speed;

    if (antPosition[idx*3] < 0 || antPosition[idx*3] >= width || antPosition[idx*3 + 1] < 0 || antPosition[idx*3 + 1] >= height) {
        antPosition[idx*3] = min(float(width - .1), max(0.0f, antPosition[idx*3]));
        antPosition[idx*3 + 1] = min(float(height - .1), max(0.0f, antPosition[idx*3 + 1]));
        const float twoPi = 6.28318530718;
        antPosition[idx*3 + 2] = randomFloatKernel(state, 0, twoPi);
    }
}

__global__ void setupCurand(curandState* state, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &state[idx]);
}

__global__ void convertColorToGradient(uchar3* image, int width, int height) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < 0 || idx >= width || idy < 0 || idy >= height) return;

	uchar3 pixel = image[idy * width + idx];

    float intensity = pixel.x;

    uchar3 newPixel = make_uchar3(intensity, intensity / 5, 0);

	image[idy * width + idx] = newPixel;
}