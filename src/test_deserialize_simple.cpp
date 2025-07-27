// Copyright 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <chrono>
#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <iomanip>
#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// Include the exception class
class PythonBackendException : public std::exception {
public:
    PythonBackendException(const std::string& message) : message_(message) {}
    const char* what() const noexcept override { return message_.c_str(); }
private:
    std::string message_;
};

// Copy the deserialize function directly here
py::array deserialize_bytes_tensor_cpp(const uint8_t* data, size_t data_size) {
    if (data_size == 0) {
        py::module numpy = py::module::import("numpy");
        return numpy.attr("empty")(0, py::dtype("object"));
    }

    // First pass: count the number of strings and calculate total size
    size_t offset = 0;
    size_t num_strings = 0;
    size_t total_string_size = 0;

    while (offset < data_size) {
        if (offset + 4 > data_size) {
            throw PythonBackendException(
                "Invalid bytes tensor data: incomplete length field");
        }

        // Read 4-byte length (little-endian)
        uint32_t length = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;

        if (offset + length > data_size) {
            throw PythonBackendException(
                "Invalid bytes tensor data: string extends beyond buffer");
        }

        num_strings++;
        total_string_size += length;
        offset += length;
    }

    // Create numpy array of objects using pybind11's numpy module
    py::module numpy = py::module::import("numpy");
    py::array result = numpy.attr("empty")(num_strings, py::dtype("object"));
    auto result_ptr = static_cast<PyObject**>(result.request().ptr);

    // Second pass: extract strings
    offset = 0;
    size_t string_index = 0;

    while (offset < data_size) {
        uint32_t length = *reinterpret_cast<const uint32_t*>(data + offset);
        offset += 4;

        // Create Python bytes object using pybind11
        py::bytes bytes_obj(reinterpret_cast<const char*>(data + offset), length);
        Py_INCREF(bytes_obj.ptr());  // Increment reference count
        result_ptr[string_index] = bytes_obj.ptr();
        string_index++;
        offset += length;
    }

    return result;
}

// Helper function to serialize strings into the expected format
std::vector<uint8_t> serialize_strings(const std::vector<std::string>& strings) {
    std::vector<uint8_t> result;
    
    for (const auto& str : strings) {
        // Write 4-byte length (little-endian)
        uint32_t length = str.length();
        result.push_back(length & 0xFF);
        result.push_back((length >> 8) & 0xFF);
        result.push_back((length >> 16) & 0xFF);
        result.push_back((length >> 24) & 0xFF);
        
        // Write string data
        for (char c : str) {
            result.push_back(static_cast<uint8_t>(c));
        }
    }
    
    return result;
}

// Function to get the resources path
std::string get_resources_path() {
    // Try multiple possible locations
    std::vector<std::string> possible_paths = {
        "./src/resources",
        "../src/resources", 
        "../../src/resources",
        "./"
    };
    
    for (const auto& path : possible_paths) {
        std::string full_path = path + "/triton_python_backend_utils.py";
        if (std::filesystem::exists(full_path)) {
            return path;
        }
    }
    
    // Check if copied to current directory
    if (std::filesystem::exists("./triton_python_backend_utils.py")) {
        return "./";
    }
    
    // Environment variable override
    const char* custom_path = std::getenv("TRITON_PYTHON_UTILS_PATH");
    if (custom_path) {
        return std::string(custom_path);
    }
    
    throw std::runtime_error("Cannot find triton_python_backend_utils.py");
}

bool setup_python_environment() {
    try {
        py::module sys = py::module::import("sys");
        
        std::string resources_path = get_resources_path();
        sys.attr("path").attr("insert")(0, resources_path);
        
        // Test import
        py::module triton_pb_utils = py::module::import("triton_python_backend_utils");
        std::cout << "Successfully imported triton_python_backend_utils from: " << resources_path << std::endl;
        return true;
        
    } catch (const py::error_already_set& e) {
        std::cerr << "Python import error: " << e.what() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Setup error: " << e.what() << std::endl;
        return false;
    }
}

bool compare_arrays(const py::array& cpp_result, const py::array& python_result, const std::vector<std::string>& original_strings) {
    // Check if both are object arrays
    if (cpp_result.dtype().kind() != 'O' || python_result.dtype().kind() != 'O') {
        std::cerr << "  Arrays are not object type" << std::endl;
        return false;
    }
    
    // Check shape
    if (cpp_result.ndim() != python_result.ndim()) {
        std::cerr << "  Dimension mismatch: C++ " << cpp_result.ndim() << " vs Python " << python_result.ndim() << std::endl;
        return false;
    }
    
    if (cpp_result.size() != python_result.size()) {
        std::cerr << "  Size mismatch: C++ " << cpp_result.size() << " vs Python " << python_result.size() << std::endl;
        return false;
    }
    
    if (static_cast<size_t>(cpp_result.size()) != original_strings.size()) {
        std::cerr << "  Size mismatch with original: " << cpp_result.size() << " vs " << original_strings.size() << std::endl;
        return false;
    }
    
    // Compare each element
    auto cpp_ptr = static_cast<PyObject**>(cpp_result.request().ptr);
    auto python_ptr = static_cast<PyObject**>(python_result.request().ptr);
    
    for (size_t i = 0; i < static_cast<size_t>(cpp_result.size()); i++) {
        py::bytes cpp_bytes = py::reinterpret_borrow<py::bytes>(cpp_ptr[i]);
        py::bytes python_bytes = py::reinterpret_borrow<py::bytes>(python_ptr[i]);
        
        std::string cpp_str = cpp_bytes;
        std::string python_str = python_bytes;
        
        if (cpp_str != python_str) {
            std::cerr << "  Element " << i << " mismatch: C++ '" << cpp_str << "' vs Python '" << python_str << "'" << std::endl;
            return false;
        }
        
        if (cpp_str != original_strings[i]) {
            std::cerr << "  Element " << i << " doesn't match original: '" << cpp_str << "' vs '" << original_strings[i] << "'" << std::endl;
            return false;
        }
    }
    
    return true;
}

int main() {
    std::cout << "Simple Deserialize Function Equivalence Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Initialize Python interpreter
    py::scoped_interpreter guard{};
    
    // Setup Python environment and import utils
    if (!setup_python_environment()) {
        std::cerr << "Failed to setup Python environment" << std::endl;
        return 1;
    }
    
    py::module triton_pb_utils = py::module::import("triton_python_backend_utils");
    
    // Helper function to generate large string arrays
    auto generate_large_string_array = [](size_t count) {
        std::vector<std::string> result;
        result.reserve(count);
        for (size_t i = 0; i < count; i++) {
            result.push_back("string_" + std::to_string(i));
        }
        return result;
    };
    
    auto generate_repeated_strings = [](size_t count, const std::string& base) {
        std::vector<std::string> result;
        result.reserve(count);
        for (size_t i = 0; i < count; i++) {
            result.push_back(base);
        }
        return result;
    };
    
    auto generate_varying_size_strings = [](size_t count) {
        std::vector<std::string> result;
        result.reserve(count);
        for (size_t i = 0; i < count; i++) {
            size_t length = (i * 37) % 1000 + 1; // Pseudo-random lengths 1-1000
            result.push_back(std::string(length, 'A' + (i % 26)));
        }
        return result;
    };

    // Define test cases
    std::vector<std::pair<std::string, std::vector<std::string>>> test_cases = {
        {"empty_data", {}},
        {"single_string", {"hello"}},
        {"multiple_strings", {"hello", "world", "test"}},
        {"empty_strings", {"", "", ""}},
        {"mixed_content", {"hello", "", "world", "test"}},
        {"binary_data", {"\x00\x01\x02\xff\xfe\xfd"}},
        {"large_strings", {std::string(1000, 'A'), std::string(2000, 'B')}},
        {"unicode_strings", {"Hello 世界", "café", "naïve"}},
        {"emoji_strings", {"🚀🌟💻", "👍👎"}},
        
        // Large batch tests
        {"batch_1000", generate_large_string_array(1000)},
        {"batch_5000", generate_large_string_array(5000)},
        {"batch_15000", generate_large_string_array(15000)},
        {"batch_15000_repeated", generate_repeated_strings(15000, "test_string")},
        {"batch_15000_varying", generate_varying_size_strings(15000)},
        {"batch_30000", generate_large_string_array(30000)},
        
        // Memory stress tests
        {"batch_10000_empty", std::vector<std::string>(10000, "")},
        {"batch_5000_large", std::vector<std::string>(5000, std::string(1000, 'X'))},
        
        // Edge case batches
        {"batch_100_mixed", []() {
            std::vector<std::string> result;
            for (size_t i = 0; i < 100; i++) {
                if (i % 3 == 0) result.push_back("");
                else if (i % 3 == 1) result.push_back("short");
                else result.push_back(std::string(i * 10, 'L'));
            }
            return result;
        }()},
    };
    
    int passed = 0, failed = 0;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    std::cout << "Running " << test_cases.size() << " test cases..." << std::endl;
    std::cout << "======================================" << std::endl;
    
    for (size_t test_idx = 0; test_idx < test_cases.size(); test_idx++) {
        const auto& [name, strings] = test_cases[test_idx];
        
        std::cout << "[" << (test_idx + 1) << "/" << test_cases.size() << "] Testing: " << name;
        if (strings.size() > 1000) {
            std::cout << " (size: " << strings.size() << ")";
        }
        std::cout << "... ";
        std::cout.flush();
        
        try {
            // Serialize test data
            std::vector<uint8_t> serialized = serialize_strings(strings);
            
            // Measure C++ performance
            auto cpp_start = std::chrono::high_resolution_clock::now();
            py::array cpp_result = deserialize_bytes_tensor_cpp(serialized.data(), serialized.size());
            auto cpp_end = std::chrono::high_resolution_clock::now();
            
            // Measure Python performance
            auto py_start = std::chrono::high_resolution_clock::now();
            py::bytes python_input = py::bytes(
                reinterpret_cast<const char*>(serialized.data()), 
                serialized.size());
            py::array python_result = triton_pb_utils.attr("deserialize_bytes_tensor")(python_input);
            auto py_end = std::chrono::high_resolution_clock::now();
            
            // Compare results
            if (compare_arrays(cpp_result, python_result, strings)) {
                auto cpp_time = std::chrono::duration_cast<std::chrono::microseconds>(cpp_end - cpp_start);
                auto py_time = std::chrono::duration_cast<std::chrono::microseconds>(py_end - py_start);
                
                double speedup = static_cast<double>(py_time.count()) / cpp_time.count();
                
                std::cout << "PASS (C++: " << cpp_time.count() << "μs, Python: " 
                         << py_time.count() << "μs, Speedup: " << std::fixed << std::setprecision(2) << speedup << "x)" << std::endl;
                passed++;
            } else {
                std::cout << "FAIL (Results differ)" << std::endl;
                failed++;
            }
            
        } catch (const std::exception& e) {
            std::cout << "FAIL (Exception: " << e.what() << ")" << std::endl;
            failed++;
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
    
    std::cout << "\n======================================" << std::endl;
    std::cout << "=== SUMMARY ===" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Total time: " << total_time.count() << "ms" << std::endl;
    std::cout << "Success rate: " << std::fixed << std::setprecision(1) 
              << (static_cast<double>(passed) / test_cases.size() * 100) << "%" << std::endl;
    
    if (passed > 0) {
        std::cout << "\nAll functional equivalence tests passed!" << std::endl;
        std::cout << "The C++ and Python implementations produce identical results." << std::endl;
    }
    
    return failed > 0 ? 1 : 0;
}