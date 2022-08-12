#ifndef CONSOLE_DEBUG_H
#define CONSOLE_DEBUG_H

#include <iostream>
#include <windows.h>
#include <ios>

#define DEBUGGER_TIMER_ID 666
#define DEBUGGER_TIMER_INTERVAL_MILI 100

class Debugger {
    public: 
        static void startDebugger(HWND m_hwnd);
        static void stopDebugger();
        static void bindTopic(int topicId, const char* topic);
        static void updateTopic(int topicId, char* line);

    private:
        static void AdjustConsoleBuffer(int16_t minLength);
        static void periodicCallback(HWND unnamedParam1, UINT unnamedParam2, UINT_PTR unnamedParam3, DWORD unnamedParam4);
        static HANDLE stdOut;
        static CHAR_INFO buffer1[100*25];
        static CHAR_INFO buffer2[100*25];
        static CHAR_INFO buffer3[100*25];
        static CHAR_INFO* buffers[3];
        static std::atomic<unsigned int> syncBitSequence;
        static std::atomic<int> usedLines;
        static int topicIdToIndexMap[25];
        static char topicIdToTopicMap[50*25];
};

#endif