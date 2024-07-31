#include <iostream>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"

#include <thread>
#include <chrono>

#include "computeShader.cuh"

using namespace cv;
using namespace std;

float randomFloat2(float min, float max) {
	float scale = static_cast<float>(rand()) / RAND_MAX;
	return min + scale * (max - min);
}

// Draw the ants on the image
Mat drawAnts(Mat image, float antPosition[][3], int numberOfAnts) {
	for (int i = 0; i < numberOfAnts; i++) {
		int x = antPosition[i][0];
		int y = antPosition[i][1];

		image.at<Vec3b>(y, x) = Vec3b(255, 255, 255);
	}
	return image;
}

Mat fadeImage(Mat image) {
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			Vec3b pixel = image.at<Vec3b>(i, j);
			if (pixel[0] == 0 && pixel[1] == 0 && pixel[2] == 0) {
				continue;
			}
			pixel *= .99;
			image.at<Vec3b>(i, j) = pixel;
		}
	}
	return image;
}

// Update the position of the ants
static void updateAntPosition(float antPosition[][3], int numberOfAnts) {
	const float stepSize = 1;
	const float angleChange = 0.1;

	for (int i = 0; i < numberOfAnts; i++) {
		// antPosition[i][2] += randomFloat(-angleChange, angleChange);
		antPosition[i][0] += stepSize * cos(antPosition[i][2]);
		antPosition[i][1] += stepSize * sin(antPosition[i][2]);

		if (antPosition[i][0] < 0 || antPosition[i][0] >= width || antPosition[i][1] < 0 || antPosition[i][1] >= height) {
			antPosition[i][0] = std::min(float(width - .1), std::max(0.0f, antPosition[i][0]));
			antPosition[i][1] = std::min(float(height - .1), std::max(0.0f, antPosition[i][1]));
			antPosition[i][2] = randomFloat2(0, twoPi);
		}
	}
}