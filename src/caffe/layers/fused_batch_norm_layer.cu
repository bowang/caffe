#ifdef USE_CUDNN
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/fused_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FusedBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                                             const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* scale_data = this->blobs_[2]->gpu_data();
  const Dtype* bias_data = this->blobs_[3]->gpu_data();
  const double epsilon = max(this->eps_, CUDNN_BN_MIN_EPSILON);

  Dtype* save_mean = save_mean_.mutable_gpu_data();
  Dtype* save_inv_var = save_inv_var_.mutable_gpu_data();

  if (this->phase_ == TRAIN) {
    // Training mode.
    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        handle_,
        mode_,
        cudnn::dataType<Dtype>::one,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_,
        bottom_data,
        top_desc_,
        top_data,
        scale_bias_mean_var_desc_,
        scale_data,
        bias_data,
        1 - this->moving_average_fraction_,
        this->blobs_[0]->mutable_gpu_data(),  // mean
        this->blobs_[1]->mutable_gpu_data(),  // variance
        epsilon,
        save_mean,
        save_inv_var));
  } else {
    // Testing mode.
    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        handle_,
        mode_,
        cudnn::dataType<Dtype>::one,
        cudnn::dataType<Dtype>::zero,
        bottom_desc_,
        bottom_data,
        top_desc_,
        top_data,
        scale_bias_mean_var_desc_,
        scale_data,
        bias_data,
        this->blobs_[0]->gpu_data(),  // mean
        this->blobs_[1]->gpu_data(),  // variance
        epsilon));
  }
}

template <typename Dtype>
void FusedBatchNormLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const Dtype* save_mean = save_mean_.gpu_data();
  const Dtype* save_inv_var = save_inv_var_.gpu_data();
  const Dtype* scale_data = this->blobs_[2]->gpu_data();
  const double epsilon = max(this->eps_, CUDNN_BN_MIN_EPSILON);

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  Dtype* scale_diff = this->blobs_[2]->mutable_gpu_diff();
  Dtype* bias_diff = this->blobs_[3]->mutable_gpu_diff();

  // call Batch Normalization Backward
  CUDNN_CHECK(cudnnBatchNormalizationBackward(
      handle_,
      mode_,
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::zero,
      cudnn::dataType<Dtype>::one,
      cudnn::dataType<Dtype>::one,
      bottom_desc_,
      bottom_data,
      top_desc_,
      top_diff,
      bottom_desc_,
      bottom_diff,
      scale_bias_mean_var_desc_,
      scale_data,
      scale_diff,
      bias_diff,
      epsilon,
      save_mean,
      save_inv_var));
}

INSTANTIATE_LAYER_GPU_FUNCS(FusedBatchNormLayer);

}  // namespace caffe
#endif
