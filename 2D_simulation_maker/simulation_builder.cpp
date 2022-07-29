#include "simulation_builder.h"

void SimulationBuilder::addParticle(int x, int y) {
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            particles.push_back(D2D1::Ellipse(D2D1::Point2F(x+2.5*i*RADIUS, y+2.5*j*RADIUS), RADIUS, RADIUS));
        }
    }
    InvalidateRect(m_hwnd, NULL, FALSE);
}

void SimulationBuilder::doNothing(int x, int y) {
    // do nothing
}

void SimulationBuilder::startLine(int x, int y){
    lines.push_back(std::make_pair(D2D1::Point2F(x, y), D2D1::Point2F(x, y)));
}

void SimulationBuilder::moveLine(int x, int y){
    lines.back().second = D2D1::Point2F(x, y);
    InvalidateRect(m_hwnd, NULL, FALSE);
}

void SimulationBuilder::mouseEvent(int x, int y, int event_type) {
    current_state = transition_table[current_state][event_type];
    (this->*action_table[current_state])(x, y);
}

std::vector<D2D1_ELLIPSE>& SimulationBuilder::getParticles(){
    return particles;
}

std::vector<std::pair<D2D1_POINT_2F, D2D1_POINT_2F>>& SimulationBuilder::getLines(){
    return lines;
}

void SimulationBuilder::keyboardEvent(short key){
    switch(key){
        case 'A':
            storeAndClear();
            break;
        default:
            break;
    }
}

void SimulationBuilder::storeAndClear(){
    std::string file_content = "PARTICLES:\n";
    for(D2D1_ELLIPSE particle : particles){
        file_content += std::to_string((int)particle.point.x) + " " + std::to_string((int)particle.point.y) + "\n";
    }
    file_content += "LINES:\n";
    for(std::pair<D2D1_POINT_2F, D2D1_POINT_2F> line : lines){
        file_content += std::to_string((int)line.first.x) + " " + std::to_string((int)line.first.y) + " " + std::to_string((int)line.second.x) + " " + std::to_string((int)line.second.y) + "\n";
    }
    HANDLE hFile;
    char* dataBuffer = &file_content[0];
    DWORD dwBytesToWrite = file_content.size();
    DWORD dwBytesWritten = 0;
    hFile = CreateFileA("../simulation_layout/simulation2D.txt", GENERIC_WRITE, 0, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);

    if (hFile == INVALID_HANDLE_VALUE)
    { 
        //TODO: error handling
        return;
    }

    BOOL bErrorFlag = WriteFile(hFile, dataBuffer, dwBytesToWrite, &dwBytesWritten, NULL);

    if (FALSE == bErrorFlag)
    {
        //TODO: error handling
    }
    else{
        if (dwBytesWritten != dwBytesToWrite)
        {
            //TODO: error handling
        }
        else
        {
            //TODO: success handling
        }
    }

    CloseHandle(hFile);

    particles.clear();
    lines.clear();
    current_state = 0;
    InvalidateRect(m_hwnd, NULL, FALSE);
}