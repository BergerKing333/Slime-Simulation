#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <thread>
#include <chrono>

#include "computeShader.cuh"

using namespace cv;
using namespace std;

struct ant {
	float x;
	float y;
	float angle;

	ant(int x, int y, int angle) : x(x), y(y), angle(angle) {}
	ant() : x(0), y(0), angle(0) {}
};

const int targetFPS = 50;
const std::chrono::milliseconds frameDuration(1000 / targetFPS);

float randomFloat(float min, float max) {
	float scale = static_cast<float>(rand()) / RAND_MAX;
	return min + scale * (max - min);
}


int main() {
	// Define constants
	Mat image = Mat::zeros(width, height, CV_8UC1);
	
	ant* antPosition = new ant[numberOfAnts];

	// Setup image on GPU
	uchar1* gpuImage;
	uchar1* tempImage;
	size_t size = width * height * sizeof(uchar);
	cudaMalloc(&gpuImage, size);
	cudaMemcpy(gpuImage, image.data, size, cudaMemcpyHostToDevice);
	cudaMalloc(&tempImage, size);
	cudaMemcpy(tempImage, image.data, size, cudaMemcpyHostToDevice);
	dim3 blockSize(32, 32);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

	int antBlockSize = 1024;
	int antGridSize((numberOfAnts + antBlockSize - 1) / antBlockSize);

	// Initialize the ants, give them a random position and a random direction
	/*for (int i = 0; i < numberOfAnts; i++) {
		antPosition[i][0] = randomFloat(0, width);
		antPosition[i][1] = randomFloat(0, height);
		antPosition[i][2] = randomFloat(0, twoPi);
	}*/

	for (int i = 0; i < numberOfAnts; i++) {
		float angle = randomFloat(0, twoPi);
		float radius = randomFloat(0, 100);
		antPosition[i].x = width / 2 + radius * cos(angle);
		antPosition[i].y = height / 2 + radius * sin(angle);
		antPosition[i].angle = angle;
	}

	//Set up ant positions on GPU
	float* flattenedAntPositions = new float[numberOfAnts * 3];
	for (int i = 0; i < numberOfAnts; ++i) {
		flattenedAntPositions[i * 3] = antPosition[i].x;
		flattenedAntPositions[i * 3 + 1] = antPosition[i].y;
		flattenedAntPositions[i * 3 + 2] = antPosition[i].angle;
	}
	float* d_antPositions;
	cudaMalloc(&d_antPositions, numberOfAnts * 3 * sizeof(float));
	cudaMemcpy(d_antPositions, flattenedAntPositions, numberOfAnts * 3 * sizeof(float), cudaMemcpyHostToDevice);

	curandState* curandStates;
	cudaMalloc(&curandStates, numberOfAnts * sizeof(curandState));
	setupCurand<<<antGridSize, antBlockSize>>>(curandStates, 120938123);

	// Frame Rendering Loop
	while (true) {
		auto start = std::chrono::high_resolution_clock::now();
		updateAntPositionKernel<<<antGridSize, antBlockSize>>>(curandStates, d_antPositions, numberOfAnts, width, height, gpuImage);
		drawAntsKernel<<<antGridSize, antBlockSize>>>(gpuImage, width, height, d_antPositions, numberOfAnts);
		diffuseKernel << <gridSize, blockSize >> > (gpuImage, tempImage, width, height);
		
		gpuImage = tempImage;
		
		// fadeImageKernel<<<gridSize, blockSize>>>(gpuImage, width, height);
		// convertColorToGradient << <gridSize, blockSize >> > (tempImage, width, height);
		cudaMemcpy(image.data, tempImage, size, cudaMemcpyDeviceToHost);
		// gpuImage = tempImage;

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsedTime = end - start;
		std::cout << "Actual FPS: " << 1000 / elapsedTime.count() << std::endl;
		std::this_thread::sleep_for(frameDuration - elapsedTime);

		cv::imshow("Slime Simulation", image);
		if (cv::waitKey(1) == 'q') {
			break;
		}


		
		// cudaDeviceSynchronize();
		// gpuImage.download(image);
		// image = drawAnts(image, antPosition, numberOfAnts);
		// image = fadeImage(image);

		
	}
}