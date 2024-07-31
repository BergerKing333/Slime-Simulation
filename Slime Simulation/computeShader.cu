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

__device__ float sense(float* antPosition, float angleOffset, uchar3* image, int width, int height) {
    float sensorAngle = antPosition[2] + angleOffset;
    float2 sensorDir = make_float2(cos(sensorAngle), sin(sensorAngle));
    int2 sensorCenter = make_int2(antPosition[0] + sensorDir.x * sensorOffsetDst, antPosition[1] + sensorDir.y * sensorOffsetDst);
    float sum = 0;
    for (int offsetX = -sensorSize; offsetX <= sensorSize; ++offsetX) {
		for (int offsetY = -sensorSize; offsetY <= sensorSize; ++offsetY) {
			int2 pos = make_int2(sensorCenter.x + offsetX, sensorCenter.y + offsetY);

            if (pos.x < 0 || pos.x >= width || pos.y < 0 || pos.y >= height) continue;
            uchar3 pixel = image[pos.y * width + pos.x];
            sum += (pixel.x + pixel.y + pixel.z) / 3.0f;
		}
	}

    return sum;
}

__device__ float randomFloatKernel(curandState* state, float min, float max) {
    float scale = curand_uniform(state);
    return min + scale * (max - min);
}

__device__ uchar3 lerp(uchar3 a, uchar3 b, float t) {
    uchar3 result;
    result.x = static_cast<unsigned char>(a.x + t * (b.x - a.x));
    result.y = static_cast<unsigned char>(a.y + t * (b.y - a.y));
    result.z = static_cast<unsigned char>(a.z + t * (b.z - a.z));
    return result;
}

__global__ void fadeImageKernel(uchar3* image, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= rows || y >= cols) return;

    // Calculate fade factor (constant for all pixels)
    

    // Apply fade to pixel
    uchar3& pixel = image[y * rows + x];
    if (pixel.x == 0 && pixel.y == 0 && pixel.z == 0) return;
    pixel.x = static_cast<unsigned char>(pixel.x * fadeFactor);
    pixel.y = static_cast<unsigned char>(pixel.y * fadeFactor);
    pixel.z = static_cast<unsigned char>(pixel.z * fadeFactor);
}

__global__ void diffuseKernel(uchar3* image, uchar3* imageOutput, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < 0 || idx >= width || idy < 0 || idy >= height) return;

    float3 sum = make_float3(0, 0, 0);
    float3 originalPixel = make_float3(image[idy * width + idx].x, image[idy * width + idx].y, image[idy * width + idx].z);

    for (int offsetX = -1; offsetX <= 1; ++offsetX) {
        for (int offsetY = -1; offsetY <= 1; ++offsetY) {
            int x = min(width - 1, max(0, idx + offsetX));
            int y = min(height - 1, max(0, idy + offsetY));
            uchar3 pixel = image[y * width + x];
            sum.x += pixel.x;
            sum.y += pixel.y;
            sum.z += pixel.z;
        }
    }
    sum.x /= 9;
    sum.y /= 9;
    sum.z /= 9;
    float3 blurredCol = make_float3(originalPixel.x * (1 - diffuseRate) + sum.x * diffuseRate, originalPixel.y * (1 - diffuseRate) + sum.y * diffuseRate, originalPixel.z * (1 - diffuseRate) + sum.z * diffuseRate);
    blurredCol.x = max(0.0f, blurredCol.x - decayRate);
    blurredCol.y = max(0.0f, blurredCol.y - decayRate);
    blurredCol.z = max(0.0f, blurredCol.z - decayRate);
    imageOutput[idy * width + idx] = make_uchar3(blurredCol.x, blurredCol.y, blurredCol.z);
}

__global__ void blurImageKernel(uchar3* imageInput, uchar3* imageOutput, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < 0 || idx >= width || idy < 0 || idy >= height) return;

    uchar3 originalPixel = imageInput[idy * width + idx];
    uchar3 sum = make_uchar3(0, 0, 0);

    for (int offsetX = -1; offsetX <= 1; ++offsetX) {
		for (int offsetY = -1; offsetY <= 1; ++offsetY) {
			int x = min(width - 1, max(0, idx + offsetX));
			int y = min(height - 1, max(0, idy + offsetY));
			uchar3 pixel = imageInput[y * width + x];
			sum.x += pixel.x;
			sum.y += pixel.y;
			sum.z += pixel.z;
		}
	}
    sum.x /= 9;
    sum.y /= 9;
    sum.z /= 9;
    // uchar3 diffused_value = sum;
    uchar3 diffused_value = lerp(originalPixel, sum, lerpRate);
    imageOutput[idy * width + idx] = make_uchar3(diffused_value.x, diffused_value.y, diffused_value.z);
}

__global__ void drawAntsKernel(uchar3* image, int width, int height, float* antPositions, int numAnts) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAnts) return;
    int x = static_cast<int>(antPositions[idx * 3]);
    int y = static_cast<int>(antPositions[idx * 3 + 1]);
    image[y * width + x] = make_uchar3(255, 100, 0);
}

__global__ void updateAntPositionKernel(curandState* state, float* antPosition, int numAnts, int width, int height, uchar3* image) {
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