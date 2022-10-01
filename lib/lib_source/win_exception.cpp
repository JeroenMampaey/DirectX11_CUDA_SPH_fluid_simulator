#include "win_exception.h"
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