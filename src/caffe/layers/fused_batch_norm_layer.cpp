#ifdef USE_CUDNN

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/fused_batch_norm_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
void FusedBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                            const vector<Blob<Dtype>*>& top) {
  const BatchNormParameter &param = this->layer_param_.batch_norm_param();
  moving_average_fraction_ = param.moving_average_fraction();
  eps_ = param.eps();

  CUDNN_CHECK(cudnnCreate(&handle_));
  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createTensor4dDesc<Dtype>(&scale_bias_mean_var_desc_);

  // Currently only SPATIAL mode is supported (most commonly used mode).
#if CUDNN_VERSION_MIN(7, 0, 0)
  mode_ = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;
#else
  mode_ = CUDNN_BATCHNORM_SPATIAL;
#endif

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    int channels = bottom[0]->channels();
    this->blobs_.resize(4);
    this->blobs_[0].reset(new Blob<Dtype>(1, channels, 1, 1));  // mean
    this->blobs_[1].reset(new Blob<Dtype>(1, channels, 1, 1));  // variance
    this->blobs_[2].reset(new Blob<Dtype>(1, channels, 1, 1));  // scale
    this->blobs_[3].reset(new Blob<Dtype>(1, channels, 1, 1));  // bias
    // Initialize scale to 1.
    caffe_set(this->blobs_[2]->count(),
              Dtype(1),
              this->blobs_[2]->mutable_cpu_data());
    // Initialize bias to 0.
    caffe_set(this->blobs_[3]->count(),
              Dtype(0),
              this->blobs_[3]->mutable_cpu_data());
  }

  handles_setup_ = true;
}

template <typename Dtype>
void FusedBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);

  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_,
                                bottom[0]->num(),
                                bottom[0]->channels(),
                                bottom[0]->height(),
                                bottom[0]->width());
  cudnn::setTensor4dDesc<Dtype>(&top_desc_,
                                top[0]->num(),
                                top[0]->channels(),
                                top[0]->height(),
                                top[0]->width());
  // Setup aux tensors for caching mean & inv_var from fwd to bwd pass.
  int channels = bottom[0]->channels();
  save_mean_.Reshape(1, channels, 1, 1);
  save_inv_var_.Reshape(1, channels, 1, 1);
  CUDNN_CHECK(cudnnDeriveBNTensorDescriptor(scale_bias_mean_var_desc_,
                                            bottom_desc_,
                                            mode_));
}

template <typename Dtype>
FusedBatchNormLayer<Dtype>::~FusedBatchNormLayer() {
  if (!handles_setup_) return;
  cudnnDestroyTensorDescriptor(bottom_desc_);
  cudnnDestroyTensorDescriptor(top_desc_);
  cudnnDestroyTensorDescriptor(scale_bias_mean_var_desc_);
  cudnnDestroy(handle_);
}

#ifdef CPU_ONLY
STUB_GPU(FusedBatchNormLayer);
#endif

INSTANTIATE_CLASS(FusedBatchNormLayer);
REGISTER_LAYER_CLASS(FusedBatchNorm);

}  // namespace caffe
#endif
