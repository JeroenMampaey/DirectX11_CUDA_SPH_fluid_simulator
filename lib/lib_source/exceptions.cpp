#include "exceptions.h"
#include <sstream>

WinException::WinException(int line, const char* file) noexcept
	:
	line( line ),
	file( file )
{}

const char* WinException::what() const noexcept
{
	std::ostringstream oss;
	oss << GetType() << std::endl
		<< GetOriginString();
	whatBuffer = oss.str();
	return whatBuffer.c_str();
}

const char* WinException::GetType() const noexcept
{
	return "WinException";
}

int WinException::GetLine() const noexcept
{
	return line;
}

const std::string& WinException::GetFile() const noexcept
{
	return file;
}

std::string WinException::GetOriginString() const noexcept
{
	std::ostringstream oss;
	oss << "[File] " << file << std::endl
		<< "[Line] " << line;
	return oss.str();
}

std::string Exception::TranslateErrorCode(HRESULT hr) noexcept
{
	char* pMsgBuf = nullptr;
	// windows will allocate memory for err string and make our pointer point to it
	const DWORD nMsgLen = FormatMessageA(
		FORMAT_MESSAGE_ALLOCATE_BUFFER |
		FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
		nullptr,hr,MAKELANGID( LANG_NEUTRAL,SUBLANG_DEFAULT ),
		reinterpret_cast<LPSTR>(&pMsgBuf),0,nullptr
	);
	// 0 string length returned indicates a failure
	if( nMsgLen == 0 )
	{
		return "Unidentified error code";
	}
	// copy error string from windows-allocated buffer to std::string
	std::string errorString = pMsgBuf;
	// free windows buffer
	LocalFree( pMsgBuf );
	return errorString;	
}

HrException::HrException( int line,const char* file,HRESULT hr ) noexcept : Exception( line,file ), hr( hr )
{}

const char* HrException::what() const noexcept{
	std::ostringstream oss;
	oss << GetType() << std::endl
		<< "[Error Code] 0x" << std::hex << std::uppercase << GetErrorCode()
		<< std::dec << " (" << (unsigned long)GetErrorCode() << ")" << std::endl
		<< "[Description] " << GetErrorDescription() << std::endl
		<< GetOriginString();
	whatBuffer = oss.str();
	return whatBuffer.c_str();
}

const char* HrException::GetType() const noexcept{
	return "Window Hr Exception";
}

HRESULT HrException::GetErrorCode() const noexcept{
	return hr;
}

std::string HrException::GetErrorDescription() const noexcept{
	return Exception::TranslateErrorCode( hr );
}

const char* DeviceRemovedException::GetType() const noexcept{
	return "Graphics Exception (DXGI_ERROR_DEVICE_REMOVED)";
}