#include "console_debug.h"

HANDLE Debugger::stdOut = NULL;

void Debugger::AdjustConsoleBuffer(int16_t minLength)
{
    // Set the screen buffer to be big enough to scroll some text
    CONSOLE_SCREEN_BUFFER_INFO conInfo;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &conInfo);
    if (conInfo.dwSize.Y < minLength)
        conInfo.dwSize.Y = minLength;
    SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE), conInfo.dwSize);
}

void Debugger::stopDebugger()
{
    if(stdOut==NULL || stdOut==INVALID_HANDLE_VALUE) return;

    stdOut = NULL;
    FreeConsole();
}

void Debugger::startDebugger(int16_t minLength, HWND m_hwnd)
{
    // Attempt to create new console
    if(!AllocConsole()) return;

    AdjustConsoleBuffer(minLength);

    stdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if(stdOut==NULL || stdOut==INVALID_HANDLE_VALUE){
        FreeConsole();
        return;
    }

    SetConsoleTitleA("Warning: Do not close this console manually!");

    DWORD mode = 0;
    if (!GetConsoleMode(stdOut, &mode))
    {
        stopDebugger();
        return;
    }

    mode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;

    // Try to set the mode.
    if (!SetConsoleMode(stdOut, mode))
    {
        stopDebugger();
        return;
    }
    
    SetTimer(m_hwnd,
            DEBUGGER_TIMER_ID,
            DEBUGGER_TIMER_INTERVAL_MILI,
            (TIMERPROC) Debugger::periodicCallback);
}

void Debugger::periodicCallback(HWND unnamedParam1, UINT unnamedParam2, UINT_PTR unnamedParam3, DWORD unnamedParam4)
{
    DWORD written = 0;
    PCWSTR sequence = L"\x1b[2J";
    WriteConsoleW(stdOut, sequence, (DWORD)wcslen(sequence), &written, NULL);

    SetConsoleCursorPosition(stdOut, {0, 0});

    const char *message = "WARNING:\n"
        "\n\n\n"
        "This console is for debugging purposes only.\n"
        "Undefined behaviour occurs when trying to close this console!! so do not close it\n"
        "\n\n\n";
    WriteConsoleA(stdOut, message, strlen(message), &written, NULL);
}