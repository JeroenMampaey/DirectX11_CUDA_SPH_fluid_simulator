#include "console_debug.h"
#include <string>

HANDLE Debugger::stdOut = NULL;
CHAR_INFO Debugger::buffer1[100*25] = {};
CHAR_INFO Debugger::buffer2[100*25] = {};
CHAR_INFO Debugger::buffer3[100*25] = {};
CHAR_INFO* Debugger::buffers[3] = {};
std::atomic<unsigned int> Debugger::syncBitSequence = 0;
std::atomic<int> Debugger::usedLines = 0;
int Debugger::topicIdToIndexMap[25] = {};
char Debugger::topicIdToTopicMap[50*25] = {};

// Helper function to set the screen buffer to be big enough to scroll some text
void Debugger::AdjustConsoleBuffer(int16_t minLength)
{
    CONSOLE_SCREEN_BUFFER_INFO conInfo;
    GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &conInfo);
    if (conInfo.dwSize.Y < minLength)
        conInfo.dwSize.Y = minLength;
    SetConsoleScreenBufferSize(GetStdHandle(STD_OUTPUT_HANDLE), conInfo.dwSize);
}

// Function to stop the debugger (freeing the console)
void Debugger::stopDebugger()
{
    if(stdOut==NULL || stdOut==INVALID_HANDLE_VALUE) return;

    stdOut = NULL;
    FreeConsole();
}

// Function to start a debugger which opens a console that can be used to display some text
void Debugger::startDebugger(int16_t minLength, HWND m_hwnd)
{
    // Attempt to create new console
    if(!AllocConsole()) return;

    AdjustConsoleBuffer(minLength);

    // Store the handle to the new console
    stdOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if(stdOut==NULL || stdOut==INVALID_HANDLE_VALUE){
        FreeConsole();
        return;
    }

    SetConsoleTitleA("Warning: Do not close this console manually!");

    // Initialize static arrays
    for(int i=0; i<100*25; i++){
        buffer1[i].Char.AsciiChar = 0;
        buffer2[i].Char.AsciiChar = 0;
        buffer3[i].Char.AsciiChar = 0;
    }

    buffers[0] = buffer1;
    buffers[1] = buffer2;
    buffers[2] = buffer3;

    for(int i=0; i<25; i++){
        topicIdToIndexMap[i] = -1;
    }

    for(int i=0; i<25*50; i++){
        topicIdToTopicMap[i] = 0;
    }
    
    // Set up a periodic callback to update the console
    SetTimer(m_hwnd,
            DEBUGGER_TIMER_ID,
            DEBUGGER_TIMER_INTERVAL_MILI,
            (TIMERPROC) Debugger::periodicCallback);
}

// Callback to update the console
void Debugger::periodicCallback(HWND unnamedParam1, UINT unnamedParam2, UINT_PTR unnamedParam3, DWORD unnamedParam4)
{
    SMALL_RECT srctWriteRect;
    srctWriteRect.Top = 0;
    srctWriteRect.Left = 0;
    srctWriteRect.Bottom = usedLines-1;
    srctWriteRect.Right = 100;

    COORD coordBufSize;
    COORD coordBufCoord;

    coordBufSize.Y = usedLines;
    coordBufSize.X = 100;

    coordBufCoord.X = 0;
    coordBufCoord.Y = 0;

    // First clear the buffer that will be used next
    int before = syncBitSequence.load() >> 30;
    memset(buffers[(before+1) % 3], 0, 100*usedLines.load()*sizeof(CHAR_INFO));

    // Then swap the framebuffer so that all new updates will be executed in the new buffer
    unsigned int oldSyncBitSequence = syncBitSequence.fetch_add(before==2 ? 2 << 30 : 1 << 30);

    // Make sure that every thread that was busy with the old buffer has finished
    unsigned int mask = oldSyncBitSequence & (~(3 << 30));
    while(mask &= syncBitSequence.load());

    // Manually copy all entries that were not updated since the last frame
    int before_before = (before+2) % 3;
    int currentUsedLines = usedLines.load();
    for(int i=0; i<currentUsedLines; i++){
        if(buffers[before][100*i].Char.AsciiChar == 0){
            memcpy(buffers[before]+100*i, buffers[before_before]+100*i, 100*sizeof(CHAR_INFO));
        }
    }
    
    // Write the buffer to the console
    WriteConsoleOutput(stdOut,
                        buffers[before],
                        coordBufSize,
                        coordBufCoord,
                        &srctWriteRect);
}

// Function to add a new topic to the debugger, this function is not necessarily completely save in case multiple threads 
// are trying to bind some topic to the same topic id, hence I assume that the caller is responsible for making sure that
// the topic id is not already in use
void Debugger::bindTopic(int topicId, char* topic){
    if(topicId < 0 || topicId >= 25){
        //TODO error: notify user that topicId is out of bounds
        return;
    }

    if(topicIdToIndexMap[topicId] != -1){
        //TODO error: notify user that topicId is already bound
        return;
    }

    int topicLength = 0;
    for(; topic[topicLength]!='\0' && topicLength<50; topicLength++);

    if(topicLength==0 || topicLength==50){
        //TODO error: notify user that topic has an invalid length
        return;
    }

    for(int i=0; i<topicLength; i++){
        topicIdToTopicMap[topicId*50+i] = topic[i];
    }

    topicIdToIndexMap[topicId] = usedLines.fetch_add(1);
}

// Function to update a topic with the given topic id
void Debugger::updateTopic(int topicId, char* line){
    if(topicId < 0 || topicId >= 25){
        //TODO error: notify user that topicId is out of bounds
        return;
    }

    int index = topicIdToIndexMap[topicId];

    if(index==-1){
        //TODO error: notify user that topicId is not bound
        return;
    }

    // Notify the debugger that the topic is being updated
    unsigned int oldSyncBitSequence = syncBitSequence.fetch_or(1 << topicId);
    
    if(oldSyncBitSequence & (1 << topicId)){
        //TODO error: notify user that writing a value in the debugger from different threads is not allowed
        syncBitSequence.fetch_xor(1 << topicId);
        return;
    }

    int currentFrame = oldSyncBitSequence >> 30;

    // First write the topic name to the buffer
    int i=0;
    for(; topicIdToTopicMap[topicId*50+i]!='\0' && i<50; i++){
        buffers[currentFrame][100*index+i].Char.AsciiChar = topicIdToTopicMap[topicId*50+i];
        buffers[currentFrame][100*index+i].Attributes = FOREGROUND_BLUE;
    }

    if(i==50){
        //TODO error: notify user that topic has an invalid length
        syncBitSequence.fetch_xor(1 << topicId);
        return;
    }

    i++;
    buffers[currentFrame][100*index+i].Char.AsciiChar = ':';
    buffers[currentFrame][100*index+i].Attributes = FOREGROUND_BLUE;
    i++;
    buffers[currentFrame][100*index+i].Char.AsciiChar = ' ';
    buffers[currentFrame][100*index+i].Attributes = FOREGROUND_BLUE;

    // Then write the line/info to the buffer
    int lineStart = i;
    for(; line[i-lineStart]!='\0' && i<100; i++){
        buffers[currentFrame][100*index+i].Char.AsciiChar = line[i-lineStart];
        buffers[currentFrame][100*index+i].Attributes = FOREGROUND_BLUE;
    }

    if(i==100){
        //TODO error: notify user that line has an invalid length
        syncBitSequence.fetch_xor(1 << topicId);
        return;
    }

    // Notify the debugger that the topic is finished updating
    syncBitSequence.fetch_xor(1 << topicId);
}