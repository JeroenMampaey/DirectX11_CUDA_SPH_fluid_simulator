#pragma once

#include <exception>
#include <string>
#include "exports.h"
#include "windows_includes.h"

class WinException : public std::exception
{
    public:
        LIBRARY_API const char* what() const noexcept override;
        LIBRARY_API virtual const char* GetType() const noexcept;
        LIBRARY_API int GetLine() const noexcept;
        LIBRARY_API const std::string& GetFile() const noexcept;
        LIBRARY_API std::string GetOriginString() const noexcept;
    private:
        int line;
        std::string file;
    protected:
        WinException(int line, const char* file) noexcept;
        mutable std::string whatBuffer;
};