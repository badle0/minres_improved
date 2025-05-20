#include "ONNXModelPredictor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

ONNXModelPredictor::ONNXModelPredictor(const std::string& model_path, int N)
    : N_(N),
      env_(ORT_LOGGING_LEVEL_WARNING, "ONNXModelPredictor")
{
    // Configure session options.
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    // Create the session
    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

    // Use an allocator to query node info
    Ort::AllocatorWithDefaultOptions allocator;

    // Input info
    {
        auto in_name_alloc = session_->GetInputNameAllocated(0, allocator);
        std::string in_name = in_name_alloc.get();
        input_node_names_.push_back(in_name);

        Ort::TypeInfo input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_node_dims_ = input_tensor_info.GetShape();
    }

    // Output info
    {
        auto out_name_alloc = session_->GetOutputNameAllocated(0, allocator);
        std::string out_name = out_name_alloc.get();
        output_node_names_.push_back(out_name);

        Ort::TypeInfo output_type_info = session_->GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_node_dims_ = output_tensor_info.GetShape();
    }

    // Debug prints (optional)
    std::cout << "Loaded model: " << model_path << std::endl;
    std::cout << "Input name: " << input_node_names_.front() << ", dims:";
    for (auto d : input_node_dims_) std::cout << " " << d;
    std::cout << "\nOutput name: " << output_node_names_.front() << ", dims:";
    for (auto d : output_node_dims_) std::cout << " " << d;
    std::cout << std::endl;
}

ONNXModelPredictor::~ONNXModelPredictor()
{
    // session_ will be cleaned up automatically.
}

void ONNXModelPredictor::predict(const LstsquaresVector & input, LstsquaresVector & output)
{
    // Keras model uses shape [batch, N, N, N, 1].
    std::vector<int64_t> input_shape = {1, N_, N_, N_, 1};

    // Expected total elements: 1 * N * N * N * 1 = N^3.
    size_t expected_size = static_cast<size_t>(N_) * N_ * N_;
    if (static_cast<size_t>(input.size) != expected_size) {
        std::cerr << "Predict Error: input.size=" << input.size 
                  << ", but expected " << expected_size << std::endl;
        return;
    }

    // Convert input from double to float.
    std::vector<float> input_data(expected_size);
    for (size_t i = 0; i < expected_size; ++i) {
        input_data[i] = static_cast<float>(input.vals[i]);
    }

    // Create the input tensor.
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Build arrays of const char* for input and output node names.
    std::vector<const char*> input_names_cstr, output_names_cstr;
    for (const auto & s : input_node_names_)  input_names_cstr.push_back(s.c_str());
    for (const auto & s : output_node_names_) output_names_cstr.push_back(s.c_str());

    // Run inference.
    auto output_tensors = session_->Run(
        Ort::RunOptions{nullptr},
        input_names_cstr.data(),
        &input_tensor,
        1,
        output_names_cstr.data(),
        1
    );

    // Extract output data.
    float* out_data = output_tensors.front().GetTensorMutableData<float>();

    // Instead of computing out_count from dynamic dimensions, use our known shape.
    size_t out_count = static_cast<size_t>(N_) * N_ * N_;

    if (out_count != expected_size) {
        std::cerr << "Predict Error: model's output has " << out_count 
                  << " elements, expected " << expected_size << std::endl;
        return;
    }

    // Check that output vector is sized correctly.
    if (output.size != static_cast<int>(out_count)) {
        std::cerr << "Predict Error: output vector size=" << output.size 
                  << ", but model returns " << out_count << " elements.\n";
        return;
    }

    // Convert the output from float to double.
    for (size_t i = 0; i < out_count; ++i) {
        output.vals[i] = static_cast<double>(out_data[i]);
    }
}
