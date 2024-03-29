#pragma once

#include "win_exception.h"

#if __has_include(<cuda.h>)
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

class Exception : public WinException{
        using WinException::WinException;
    public:
        static std::string TranslateErrorCode(HRESULT hr) noexcept;
#if __has_include(<cuda.h>)
        static std::string TranslateErrorCode(cudaError_t err) noexcept;
#endif
};

class HrException : public Exception
{
    public:
        HrException(int line, const char* file, HRESULT hr) noexcept;
        const char* what() const noexcept override;
        const char* GetType() const noexcept override;
        HRESULT GetErrorCode() const noexcept;
        std::string GetErrorDescription() const noexcept;
    private:
        HRESULT hr;
};

class DeviceRemovedException : public HrException
{
        using HrException::HrException;
    public:
        const char* GetType() const noexcept override;
    private:
        std::string reason;
};

#if __has_include(<cuda.h>)
class CudaException : public Exception{
    public:
        CudaException(int line, const char* file, cudaError_t err) noexcept;
        const char* what() const noexcept override;
        const char* GetType() const noexcept override;
        cudaError_t GetErrorCode() const noexcept;
        std::string GetErrorDescription() const noexcept;
    private:
        cudaError_t err;
};

#define CUDA_THROW_FAILED(cudaCall) if( ( err = (cudaCall) ) != cudaSuccess ) throw CudaException( __LINE__,__FILE__,err)
#endif

#define GFX_THROW_FAILED(hrcall) if( FAILED( hr = (hrcall) ) ) throw HrException( __LINE__,__FILE__,hr )
#define CHWND_LAST_EXCEPT() HrException( __LINE__,__FILE__,GetLastError() )
#define GFX_DEVICE_REMOVED_EXCEPT(hr) DeviceRemovedException( __LINE__,__FILE__,(hr))