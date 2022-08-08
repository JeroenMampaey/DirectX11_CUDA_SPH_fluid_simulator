#ifndef CONSOLE_DEBUG_H
#define CONSOLE_DEBUG_H

#include <iostream>
#include <windows.h>
#include <ios>

void AdjustConsoleBuffer(int16_t minLength);

bool RedirectConsoleIO();

bool ReleaseConsole();

bool CreateNewConsole(int16_t minLength);

#endif