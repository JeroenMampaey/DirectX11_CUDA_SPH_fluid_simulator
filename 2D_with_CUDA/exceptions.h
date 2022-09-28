#pragma once

#include <exception>
#include <string>

#include <cuda_runtime_api.h>
#include <cuda.h>

class CudaException : public std::exception{
    public:
        CudaException(int line, const char* file, cudaError_t err) noexcept;
        const char* what() const noexcept override;
        const char* GetType() const noexcept;
        int GetLine() const noexcept;
        const std::string& GetFile() const noexcept;
        std::string GetOriginString() const noexcept;
        std::string GetErrorDescription() const noexcept;
        cudaError_t GetErrorCode() const noexcept;
        static std::string TranslateErrorCode(cudaError_t err) noexcept;
    private:
        cudaError_t err;
        int line;
        std::string file;
    protected:
        mutable std::string whatBuffer;
};

#define CUDA_THROW_FAILED(cudaCall) if( ( err = (cudaCall) ) != cudaSuccess ) throw CudaException( __LINE__,__FILE__,err)