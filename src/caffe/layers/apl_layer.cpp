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
	void APLLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& 