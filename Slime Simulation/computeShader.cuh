#ifndef COMPUTERSHADER_CUH
#define COMPUTERSHADER_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>

constexpr int sensorOffsetDst = 5;
constexpr int sensorSize = 5;
constexpr float turnSpeed = 1.2;
constexpr float lerpRate = 1.5;
constexpr float speed = 1;
constexpr float fadeFactor = .99;
constexpr float diffuseRate = .01;
constexpr float decayRate = .5;

constexpr const int width = 1200;
constexpr const int height = 1200;
constexpr const float twoPi = 2 * 3.14159265;
constexpr const int numberOfAnts = 100000;


__global__ void fadeImageKernel(uchar3* image, int rows, int cols);

__global__ void drawAntsKernel(uchar3* image, int rows, int cols, float* antPositions, int numAnts);

__global__ void updateAntPositionKernel(curandState* state, float* antPositions, int numAnts, int width, int height, uchar3* image);

__global__ void setupCurand(curandState* state, unsigned long seed);

__device__ float randomFloatKernel(curandState* state, float min, float max);

__device__ uchar3 lerp(uchar3 a, uchar3 b, float t);

__global__ void blurImageKernel(uchar3* imageInput, uchar3* imageOutput, int width, int height);

__global__ void diffuseKernel(uchar3* imageInput, uchar3* imageOutput, int width, int height);

__global__ void convertColorToGradient(uchar3* image, int width, int height);

#endif;