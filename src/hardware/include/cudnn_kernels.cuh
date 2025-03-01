/**
 * HyVERX Hardware Acceleration cuDNN Kernels Header
 * 
 * This header defines the interface for cuDNN-optimized kernels for neural network
 * operations on NVIDIA GPUs. It leverages NVIDIA's cuDNN library for high-performance
 * deep learning primitives and provides specialized implementations for neural
 * network operations and neurosymbolic integration.
 */

#ifndef HYVERX_CUDNN_KERNELS_CUH
#define HYVERX_CUDNN_KERNELS_CUH

#include <cudnn.h>
#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <string>

// Forward declarations
class CuDNNManager;

extern "C" {
    // Convolution operation using cuDNN
    cudnnStatus_t neurosymbolic_convolution_forward(
        const float* input,
        const float* filter,
        float* output,
        int batch_size,
        int in_channels,
        int in_height,
        int in_width,
        int filter_out_channels,
        int filter_height,
        int filter_width,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        cudaStream_t stream = nullptr
    );
    
    // Pooling operation using cuDNN
    cudnnStatus_t neurosymbolic_pooling_forward(
        const float* input,
        float* output,
        int batch_size,
        int channels,
        int in_height,
        int in_width,
        int window_h,
        int window_w,
        int pad_h,
        int pad_w,
        int stride_h,
        int stride_w,
        cudnnPoolingMode_t mode = CUDNN_POOLING_MAX,
        cudaStream_t stream = nullptr
    );
    
    // Activation operation using cuDNN
    cudnnStatus_t neurosymbolic_activation_forward(
        const float* input,
        float* output,
        int batch_size,
        int channels,
        int height,
        int width,
        cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU,
        double coef = 0.0,
        cudaStream_t stream = nullptr
    );
    
    // Softmax operation using cuDNN
    cudnnStatus_t neurosymbolic_softmax_forward(
        const float* input,
        float* output,
        int batch_size,
        int channels,
        int height,
        int width,
        cudnnSoftmaxAlgorithm_t algo = CUDNN_SOFTMAX_ACCURATE,
        cudnnSoftmaxMode_t mode = CUDNN_SOFTMAX_MODE_CHANNEL,
        cudaStream_t stream = nullptr
    );
    
    // Batch normalization operation using cuDNN
    cudnnStatus_t neurosymbolic_batch_norm_forward(
        const float* input,
        float* output,
        const float* scale,
        const float* bias,
        float* running_mean,
        float* running_var,
        float* save_mean,
        float* save_var,
        int batch_size,
        int channels,
        int height,
        int width,
        double epsilon = 1e-5,
        double exp_avg_factor = 0.1,
        cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL,
        bool training = true,
        cudaStream_t stream = nullptr
    );
    
    // LSTM operation using cuDNN
    cudnnStatus_t neurosymbolic_lstm_forward(
        const float* input,
        const float* hx,
        const float* cx,
        const float* weights,
        float* output,
        float* hy,
        float* cy,
        int batch_size,
        int seq_length,
        int input_size,
        int hidden_size,
        int num_layers = 1,
        bool bidirectional = false,
        cudaStream_t stream = nullptr
    );
    
    // Neurosymbolic integration: combine neural results with symbolic reasoning
    cudaError_t launch_neurosymbolic_integration(
        const float* neural_activations,
        const float* symbolic_values,
        float* integrated_output,
        int batch_size,
        int neural_size,
        int symbolic_size,
        int output_size,
        float integration_weight = 0.5f,
        cudaStream_t stream = nullptr
    );
    
    // Utility function to check cuDNN version
    void print_cudnn_version();
}

// Utility function to check if cuDNN is available
bool is_cudnn_available();

// Utility function to get cuDNN version
std::string get_cudnn_version_string();

// Utility class for managing cuDNN resources
class CuDNNHandle {
public:
    // Constructor
    CuDNNHandle();
    
    // Destructor
    ~CuDNNHandle();
    
    // Get cuDNN handle
    cudnnHandle_t get() const;
    
    // Set stream for handle
    void set_stream(cudaStream_t stream);
    
private:
    cudnnHandle_t handle_;
};

// Factory function to create tensor descriptor
cudnnTensorDescriptor_t create_tensor_descriptor(
    int n, int c, int h, int w,
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT,
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW
);

// Factory function to create filter descriptor
cudnnFilterDescriptor_t create_filter_descriptor(
    int out_channels, int in_channels, int h, int w,
    cudnnDataType_t data_type = CUDNN_DATA_FLOAT,
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW
);

// Factory function to create convolution descriptor
cudnnConvolutionDescriptor_t create_convolution_descriptor(
    int pad_h, int pad_w, int stride_h, int stride_w,
    int dilation_h = 1, int dilation_w = 1,
    cudnnDataType_t compute_type = CUDNN_DATA_FLOAT
);

// Factory function to create pooling descriptor
cudnnPoolingDescriptor_t create_pooling_descriptor(
    cudnnPoolingMode_t mode, int window_h, int window_w,
    int pad_h, int pad_w, int stride_h, int stride_w
);

// Factory function to create activation descriptor
cudnnActivationDescriptor_t create_activation_descriptor(
    cudnnActivationMode_t mode,
    cudnnNanPropagation_t nan_opt = CUDNN_NOT_PROPAGATE_NAN,
    double coef = 0.0
);

// Utility function to find best convolution algorithm
cudnnConvolutionFwdAlgo_t find_best_convolution_algorithm(
    cudnnHandle_t handle,
    const cudnnTensorDescriptor_t input_desc,
    const cudnnFilterDescriptor_t filter_desc,
    const cudnnConvolutionDescriptor_t conv_desc,
    const cudnnTensorDescriptor_t output_desc
);

#endif // HYVERX_CUDNN_KERNELS_CUH