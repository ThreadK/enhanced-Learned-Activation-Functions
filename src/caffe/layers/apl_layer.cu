
// Copyright 2014 BVLC and contributors.

#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
	__global__ void APLForwardSum(const int n, int s, const Dtype* in, Dtype* out, const Dtype* neuron_weight, const Dtype* neuron_offset, Dtype* maxs_data, int sums_, int K_) {
		CUDA_KERNEL_LOOP(index, n) {
			int exPos = ((int) index / K_) * K_;
			int exPosSums = exPos*sums_;
			int k = index % K_;
			int sumPos = k*sums_;

			if (s == 0) {
				out[index] = in[index] > 0 ? in[index] : 0;
			}
			maxs_data[exPosSums + sumPos + s] = max(-in[index] + neuron_offset[sumPos + s], Dtype(0));
			out[index] += neuron_weight[sumPos + s]*maxs_data[exPosSums + sumPos + s];
		}
	}

template <typename Dtype>
	__global__ void APLForwardSumHardcode(const int n, const Dtype* in, Dtype* out, const Dtype* neuron_weight, const Dtype* neuron_offset, Dtype* maxs_data, int sums_, int K_) {
		CUDA_KERNEL_LOOP(index, n) {
			int exPos = ((int) index / K_) * K_;
			int exPosSums = exPos*sums_;
			int k = index % K_;
			int sumPos = k*sums_;

			switch (sums_) {
				case 1 : { 
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0];
									 break;
								 }
				case 2 : {
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));
									 maxs_data[exPosSums + sumPos + 1] = max(-in[index] + neuron_offset[sumPos + 1], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0] + neuron_weight[sumPos + 1]*maxs_data[exPosSums + sumPos + 1];
									 break;
								 }
				case 3 : {
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));
									 maxs_data[exPosSums + sumPos + 1] = max(-in[index] + neuron_offset[sumPos + 1], Dtype(0));
									 maxs_data[exPosSums + sumPos + 2] = max(-in[index] + neuron_offset[sumPos + 2], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0] + neuron_weight[sumPos + 1]*maxs_data[exPosSums + sumPos + 1] + neuron_weight[sumPos + 2]*maxs_data[exPosSums + sumPos + 2];
									 break;
								 }
				case 4 : {
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));
									 maxs_data[exPosSums + sumPos + 1] = max(-in[index] + neuron_offset[sumPos + 1], Dtype(0));
									 maxs_data[exPosSums + sumPos + 2] = max(-in[index] + neuron_offset[sumPos + 2], Dtype(0));
									 maxs_data[exPosSums + sumPos + 3] = max(-in[index] + neuron_offset[sumPos + 3], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0] + neuron_weight[sumPos + 1]*maxs_data[exPosSums + sumPos + 1] + neuron_weight[sumPos + 2]*maxs_data[exPosSums + sumPos + 2] + neuron_weight[sumPos + 3]*maxs_data[exPosSums + sumPos + 3];
									 break;
								 }
				case 5 : {
									 maxs_data[exPosSums + sumPos + 0] = max(-in[index] + neuron_offset[sumPos + 0], Dtype(0));
									 maxs_data[exPosSums + sumPos + 1] = max(-in[index] + neuron_offset[sumPos + 1], Dtype(0));
									 maxs_data[exPosSums + sumPos + 2] = max(-in[index] + neuron_offset[sumPos + 2], Dtype(0));
									 maxs_data[exPosSums + sumPos + 3] = max(-in[index] + neuron_offset[sumPos + 3], Dtype(0));
									 maxs_data[exPosSums + sumPos + 4] = max(-in[index] + neuron_offset[sumPos + 4], Dtype(0));

									 Dtype inMax = in[index] > 0 ? in[index] : 0;
									 out[index] = inMax +  neuron_weight[sumPos + 0]*maxs_data[exPosSums + sumPos + 0] + neuron_weight[sumPos + 1]*maxs_data[exPosSums + sumPos + 1] + neuron_weight[sumPos + 2]*maxs_data[exPosSums + sumPos + 2] + neuron_weight[sumPos + 3]*maxs_data[exPosSums + sumPos + 3] + neuron_weight[sumPos + 4]*maxs_data[exPosSums + sumPos + 4];
									 break;
								 }
			}
		}
	}

template <typename Dtype>
	void APLLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		//Forward_cpu(bottom,top);

		//Initialize