#pragma once

#ifdef READ_FROM_LIB_HEADER
#define LIBRARY_API __declspec(dllimport)
#else
#define LIBRARY_API __declspec(dllexport)
#endif