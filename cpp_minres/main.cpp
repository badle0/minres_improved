#include <source/solver.h>  // Main sym-ildl header (header-only library)

#include <iostream>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

// Helper functions
#include "helpers/helper_functions.hpp"    // readA_sparse(...) and get_vector_from_source(...)
#include "lstsquares_example/LstsquaresVector.hpp"
#include "lstsquares_example/LstsquaresOperator.hpp"
#include "lstsquares_example/SOperator.hpp"  // CSR-based operator for your binary matrix S
#include "helpers/minressparse.hpp"        // Your standard and deep MINRES routines
#include "ONNXModelPredictor.hpp"  // ONNX predictor for deep MINRES

// --- Conversion function: Write CSRMatrix in Matrix Market format ---
// Matrix Market files are 1-indexed and have a header.
bool writeMatrixMarket(const CSRMatrix & A, const std::string & filename) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Error opening " << filename << " for writing." << std::endl;
        return false;
    }
    // Write header
    ofs << "%%MatrixMarket matrix coordinate real general" << std::endl;
    ofs << A.num_rows << " " << A.num_cols << " " << A.nnz << std::endl;
    
    // Loop through rows (convert 0-index to 1-index)
    for (int i = 0; i < A.num_rows; i++) {
        int start = A.indptr[i];
        int end = A.indptr[i+1];
        for (int j = start; j < end; j++) {
            int col = A.indices[j];
            double value = A.data[j];
            ofs << (i + 1) << " " << (col + 1) << " " << value << std::endl;
        }
    }
    ofs.close();
    return true;
}

int main()
{
    // --- Steps 1-8: Existing code using binary files ---
    std::string base_dir = "/data/hsheng/virtualenvs/minres_improved/2024minresdata/lstsquareproblem/test_dataset";
    int test_id = 3; // e.g., test case 3

    // Build file paths for binary matrix S and vector d.
    std::string s_file = base_dir + "/S_test" + std::to_string(test_id) + ".bin";
    std::string d_file = base_dir + "/d_test" + std::to_string(test_id) + ".bin";

    DataType s_dtype = DataType::Float32;  // For the S matrix
    DataType d_dtype = DataType::Float32;  // For the vector d

    std::cout << "Loading augmented matrix S from " << s_file << "...\n";
    CSRMatrix S_csr = readA_sparse(s_file, s_dtype);
    if (S_csr.num_rows != S_csr.num_cols) {
        std::cerr << "Error: loaded matrix from " << s_file
                  << " is not square (" << S_csr.num_rows << "x" << S_csr.num_cols << ")!\n";
        return 1;
    }
    int system_size = S_csr.num_rows;  // e.g., 262144
    std::cout << "Read S: " << system_size << " x " << system_size
              << ", nnz = " << S_csr.nnz << std::endl;

    LstsquaresVector d = get_vector_from_source(d_file, d_dtype);
    if (d.size != system_size) {
        std::cerr << "Error: loaded d of size " << d.size
                  << ", but expected " << system_size << std::endl;
        return 1;
    }
    std::cout << "Loaded d (size = " << d.size << ") from " << d_file << std::endl;
    double sumAbsD = 0.0;
    for (int i = 0; i < d.size; i++) {
        sumAbsD += std::fabs(d[i]);
    }
    std::cout << "Sum of absolute values of d: " << sumAbsD << std::endl;

    // Construct the operator for S (for use in your MINRES routines)
    SOperator op(S_csr);

    int max_iter = 1000;
    double tol = 1e-4;
    bool verbose = false;

    // --- Step 7: Standard MINRES ---
    {
        std::cout << "\n--- Running Standard MINRESSparse on S y = d ---\n";
        LstsquaresVector y(system_size);
        y = 0.0;  // zero initial guess

        int istop = MINRESSparse(op, y, d, max_iter, tol, verbose);
        std::cout << "Standard MINRESSparse finished, istop=" << istop << std::endl;

        // Compute residual r = d - S*y
        LstsquaresVector r(system_size);
        op.Apply(y, r);     // r = S*y
        subtract(d, r, r);  // r = d - S*y
        double rnorm = std::sqrt(InnerProduct(r, r));
        std::cout << "Residual norm after standard MINRES: " << rnorm << std::endl;
    }

    // --- Step 8: Deep MINRES with ONNX ---
    {
        std::cout << "\n--- Running Deep MINRESSparse (with ONNX) on S y = d ---\n";
        LstsquaresVector y(system_size);
        y = 0.0; // reset guess

        std::string model_path = "/data/hsheng/virtualenvs/minres_improved/cpp_minres/saved_model/model.onnx";
        ONNXModelPredictor predictor(model_path, 64);
        auto model_predict = [&predictor](const LstsquaresVector & in, LstsquaresVector & out) {
            predictor.predict(in, out);
        };

        int istop_deep = DeepMINRESSparse(op, y, d, model_predict, max_iter, tol, verbose);
        std::cout << "Deep MINRESSparse finished, istop=" << istop_deep << std::endl;

        LstsquaresVector r(system_size);
        op.Apply(y, r);    // r = S*y
        subtract(d, r, r); // r = d - S*y
        double rnorm = std::sqrt(InnerProduct(r, r));
        std::cout << "Residual norm after Deep MINRES: " << rnorm << std::endl;
    }

    // --- Step 9: Preconditioned MINRES using SYM-ILDL directly ---
    {
        std::cout << "\n--- Running Preconditioned MINRES using SYM-ILDL ---\n";
        // Convert the in-memory CSRMatrix to a Matrix Market file.
        std::string mtx_file = base_dir + "/S_test3.mtx";
        if (!writeMatrixMarket(S_csr, mtx_file)) {
            std::cerr << "Failed to write Matrix Market file to " << mtx_file << std::endl;
            return 1;
        }
        std::cout << "Converted CSR matrix to Matrix Market format: " << mtx_file << std::endl;

        // Create a sym-ildl solver object.
        symildl::solver<double, lilc_matrix<double>> ildlSolver;
        
        try {
            ildlSolver.load(mtx_file);
        } catch (const std::exception & e) {
            std::cerr << "Failed to load matrix in ILDL solver: " << e.what() << std::endl;
            return 1;
        }
        
        // Set solver parameters (mimicking the command-line example).
        ildlSolver.set_reorder_scheme("amd");
        ildlSolver.set_equil("bunch");
        ildlSolver.set_solver("minres");
        ildlSolver.set_pivot("rook");
        ildlSolver.set_message_level("statistics");
        ildlSolver.set_inplace(false);
        
        // Convert your LstsquaresVector 'd' (rhs) to std::vector<double> as expected by sym-ildl.
        std::vector<double> rhs_std(d.size);
        for (int i = 0; i < d.size; i++) {
            rhs_std[i] = d[i];
        }
        ildlSolver.set_rhs(rhs_std);
        
        // Now solve the linear system.
        ildlSolver.solve(3.0, 0.00001, 1, 1000, 1e-6);
        
        // Optionally, save the computed factors and solution.
        ildlSolver.save();
        
        std::cout << "Preconditioned MINRES using incomplete LDL^T (SYM-ILDL) complete." << std::endl;
    }
    
    std::cout << "\nDone. Exiting.\n";
    return 0;
}
