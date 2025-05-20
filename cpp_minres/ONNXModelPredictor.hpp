#ifndef ONNX_MODEL_PREDICTOR_HPP
#define ONNX_MODEL_PREDICTOR_HPP

#include <string>
#include <vector>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "LstsquaresVector.hpp"

/*!
 * @class ONNXModelPredictor
 * @brief A wrapper class around ONNXRuntime for loading and running an ONNX model.
 *
 * This class loads a given model (.onnx file) and runs inference on a vector
 * shaped as [1, N, N, N, 1] (for a 3D volume of side N and 1 channel).
 *
 * The output is assumed also to have shape [1, N, N, N, 1] (or the same total number of elements).
 * Adjust the internal logic as needed if your model has a different shape.
 */
class ONNXModelPredictor {
public:
    /*!
     * @brief Constructor
     * @param model_path Path to the .onnx file
     * @param N The 3D resolution (e.g., 64)
     */
    ONNXModelPredictor(const std::string & model_path, int N);

    //! Destructor
    ~ONNXModelPredictor();

    /*!
     * @brief Runs inference on the given input vector.
     * @param input A LstsquaresVector of size N^3 (for a single sample, shaped [1, N, N, N, 1])
     * @param output A LstsquaresVector to receive the model output (also size N^3).
     *
     * Internally:
     *  - The input is reshaped to [1, N, N, N, 1].
     *  - The model is run via ONNXRuntime.
     *  - The output is copied back into @p output as double precision.
     */
    void predict(const LstsquaresVector & input, LstsquaresVector & output);

private:
    int N_; //!< The resolution dimension (64, 128, etc.)

    //! ONNXRuntime environment and session
    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;

    /*!
     * @brief Node names and shapes
     *
     * ONNXRuntime requires us to pass arrays of const char* for Run(...).
     * We store them here as std::string and build char* arrays on the fly.
     */
    std::vector<std::string> input_node_names_;
    std::vector<std::string> output_node_names_;

    //! Shapes from the model's input and output
    std::vector<int64_t> input_node_dims_;
    std::vector<int64_t> output_node_dims_;
};

#endif // ONNX_MODEL_PREDICTOR_HPP
