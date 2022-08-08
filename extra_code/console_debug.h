#ifndef CONSOLE_DEBUG_H
#define CONSOLE_DEBUG_H

#include <iostream>
#include <windows.h>
#include <ios>

#define DEBUGGER_TIMER_ID 666
#define DEBUGGER_TIMER_INTERVAL_MILI 100

class Debugger {
    public: 
        static void startDebugger(int16_t minLength, HWND m_hwnd);
        static void stopDebugger();

    private:
        static void AdjustConsoleBuffer(int16_t minLength);
        static void periodicCallback(HWND unnamedParam1, UINT unnamedParam2, UINT_PTR unnamedParam3, DWORD unnamedParam4);
        static HANDLE stdOut;
};

#endif