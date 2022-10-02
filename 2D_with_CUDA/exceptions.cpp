#include "exceptions.h"
#include <sstream>

CudaException::CudaException(int line, const char* file, cudaError_t err) noexcept
    :
    err(err),
    line(line),
    file(file)
{}

const char* CudaException::what() const noexcept{
	std::ostringstream oss;
	oss << GetType() << std::endl
		<< "[Error Code] 0x" << std::hex << std::uppercase << GetErrorCode()
		<< std::dec << " (" << (unsigned long)GetErrorCode() << ")" << std::endl
		<< "[Description] " << GetErrorDescription() << std::endl
		<< GetOriginString();
	whatBuffer = oss.str();
	return whatBuffer.c_str();
}

const char* CudaException::GetType() const noexcept{
	return "CudaException";
}

int CudaException::GetLine() const noexcept
{
	return line;
}

const std::string& CudaException::GetFile() const noexcept{
	return file;
}

std::string CudaException::GetOriginString() const noexcept{
	std::ostringstream oss;
	oss << "[File] " << file << std::endl
		<< "[Line] " << line;
	return oss.str();
}

std::string CudaException::TranslateErrorCode(cudaError_t err) noexcept{
	const char* pMsgBuf = cudaGetErrorString(err);
	std::string errorString = pMsgBuf;
	return errorString;	
}

std::string CudaException::GetErrorDescription() const noexcept{
    return CudaException::TranslateErrorCode(err);
}

cudaError_t CudaException::GetErrorCode() const noexcept{
    return err;
}

