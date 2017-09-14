#ifndef CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
#define CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/batch_norm_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/*
 * @brief Normalizes the input to have 0-mean and/or unit (1) variance across
 *        the batch and then scales the result.
 *
 * FusedBatchNormLayer combines the original BatchNormLayer and the following
 * ScaleLayer for better performance.
 */
template <typename Dtype>
class FusedBatchNormLayer : public Layer<Dtype> {
public:
  explicit FusedBatchNormLayer(const LayerParameter& param)
      : Layer<Dtype>(param), handles_setup_(false) {}

  virtual ~FusedBatchNormLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                       const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top) {
    LOG(FATAL) << "FusedBatchNormLayer::Forward_cpu() is not implemented";
  }

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom) {
    LOG(FATAL) << "FusedBatchNormLayer::Backward_cpu() is not implemented";
  }

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& bottom);

  cudnnTensorDescriptor_t bottom_desc_, top_desc_;
  cudnnTensorDescriptor_t scale_bias_mean_var_desc_;
  cudnnBatchNormMode_t mode_;
  cudnnHandle_t handle_;
  bool handles_setup_;

  Dtype moving_average_fraction_;
  Dtype eps_;
  Blob<Dtype> save_mean_, save_inv_var_;
};
#endif

}  // namespace caffe

#endif  // CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
