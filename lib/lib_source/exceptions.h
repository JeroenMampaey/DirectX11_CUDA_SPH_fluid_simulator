#pragma once

#include <exception>
#include <string>
#include <windows.h>
#include "exports.h"

class LIBRARY_API WinException : public std::exception
{
    public:
        WinException(int line, const char* file) noexcept;
        const char* what() const noexcept override;
        virtual const char* GetType() const noexcept;
        int GetLine() const noexcept;
        const std::string& GetFile() const noexcept;
        std::string GetOriginString() const noexcept;
    private:
        int line;
        std::string file;
    protected:
        mutable std::string whatBuffer;
};

class Exception : public WinException
{
        using WinException::WinException;
    public:
        static std::string TranslateErrorCode(HRESULT hr) noexcept;
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

#define GFX_THROW_FAILED(hrcall) if( FAILED( hr = (hrcall) ) ) throw HrException( __LINE__,__FILE__,hr )
#define CHWND_LAST_EXCEPT() HrException( __LINE__,__FILE__,GetLastError() )
#define GFX_DEVICE_REMOVED_EXCEPT(hr) DeviceRemovedException( __LINE__,__FILE__,(hr))