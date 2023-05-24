// Copyright 2014 BVLC and contributors.

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/syncedmem.hpp"
#include <ctime>
#include <cstdio>
#include <string.h>
#include <locale>

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
	void APLLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
		CHECK_GE(bottom[0]->num_axes(), 2)
			<< "Number of axes of bottom blob must be >=2.";

		// Figure out the dimensions
		M_ = bottom[0]->num();
		K_ = bottom[0]->count() / bottom[0]->num();
		N_ = K_;

		sums_ = this->layer_param_.apl_param().sums();
		save_mem_ = this->layer_param_.apl_param().save_mem();

		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		} 

		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
		} else {
			this->blobs_.resize(2);

			shared_ptr<Filler<Dtype> > slope_filler;
			if (this->layer_param_.apl_param().has_slope_filler()) {
				slope_filler.reset(GetFiller<Dtype>(this->layer_param_.apl_param().slope_filler()));
			} else {
				FillerParameter slope_filler_param;
				slope_filler_param.set_type("uniform");
				slope_filler_param.set_min((Dtype) -0.5/((Dtype) sums_));