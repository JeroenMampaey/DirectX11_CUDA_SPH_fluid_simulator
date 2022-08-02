#include "simulation_builder.h"

// Add a small square of particles starting at mouseX, mouseY
void SimulationBuilder::addParticle() {
    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            particles.push_back(D2D1::Ellipse(D2D1::Point2F(mouseX+2.5*i*RADIUS, mouseY+2.5*j*RADIUS), RADIUS, RADIUS));
        }
    }
    InvalidateRect(m_hwnd, NULL, FALSE);
}

void SimulationBuilder::doNothing() {
    // do nothing
}

// Start a new line at mouseX, mouseY
void SimulationBuilder::startLine(){
    lines.push_back(std::make_pair(D2D1::Point2F(mouseX, mouseY), D2D1::Point2F(mouseX, mouseY)));
}

// Move the last point of the last line to mouseX, mouseY
void SimulationBuilder::moveLine(){
    lines.back().second = D2D1::Point2F(mouseX, mouseY);
    InvalidateRect(m_hwnd, NULL, FALSE);
}

// Start a new box at mouseX, mouseY
void SimulationBuilder::startBox(){
    boxes.push_back(std::make_pair(D2D1::RectF(mouseX, mouseY, mouseX, mouseY), D2D1::Point2F(DEFAULT_PUMP_VELOCITY, 0)));
}

// Move the "top-left point" of the last box to mouseX, mouseY (even though it's called the "top-left point", it is not necessarily top-left)
void SimulationBuilder::moveBox(){
    boxes.back().first.left = mouseX;
    boxes.back().first.top = mouseY;
    InvalidateRect(m_hwnd, NULL, FALSE);
}

// Move the velocity of the last box counter-clockwise
void SimulationBuilder::moveBoxDirectionLeft(){
    boxes.back().second = D2D1::Point2F(boxes.back().second.y, -boxes.back().second.x);
    InvalidateRect(m_hwnd, NULL, FALSE);
}

// Move the velocity of the last box clockwise
void SimulationBuilder::moveBoxDirectionRight(){
    boxes.back().second = D2D1::Point2F(-boxes.back().second.y, boxes.back().second.x);
    InvalidateRect(m_hwnd, NULL, FALSE);
}

// Increase the velocity of the last box
void SimulationBuilder::moveBoxVelocityUp(){
    boxes.back().second = D2D1::Point2F(boxes.back().second.x * 1.1, boxes.back().second.y * 1.1);
    InvalidateRect(m_hwnd, NULL, FALSE);
}

// Decrease the velocity of the last box
void SimulationBuilder::moveBoxVelocityDown(){
    boxes.back().second = D2D1::Point2F(boxes.back().second.x * (1/1.1), boxes.back().second.y * (1/1.1));
    InvalidateRect(m_hwnd, NULL, FALSE);
}

std::vector<D2D1_ELLIPSE>& SimulationBuilder::getParticles(){
    return particles;
}

std::vector<std::pair<D2D1_POINT_2F, D2D1_POINT_2F>>& SimulationBuilder::getLines(){
    return lines;
}

std::vector<std::pair<D2D1_RECT_F, D2D1_POINT_2F>>& SimulationBuilder::getBoxes(){
    return boxes;
}

// Respond to events according to an automaton
void SimulationBuilder::event(int event_type){
    current_state = transition_table[current_state][event_type];
    (this->*action_table[current_state])();
}

// Clear the screen and store the particles and boundaries in a file at "../simulation_layout/simulation2D.txt"
void SimulationBuilder::store(){
    // Convert the particles, boundaries and pumps to a string format
    std::string file_content = "PARTICLES:\n";
    for(D2D1_ELLIPSE particle : particles){
        file_content += std::to_string((int)particle.point.x) + " " + std::to_string((int)particle.point.y) + "\n";
    }
    file_content += "LINES:\n";
    for(std::pair<D2D1_POINT_2F, D2D1_POINT_2F> line : lines){
        if(line.second.x!=line.first.x || line.second.y!=line.first.y){
            file_content += std::to_string((int)line.first.x) + " " + std::to_string((int)line.first.y) + " " + std::to_string((int)line.second.x) + " " + std::to_string((int)line.second.y) + "\n";
        }
    }
    file_content += "PUMPS:\n";
    for(std::pair<D2D1_RECT_F, D2D1_POINT_2F> box : boxes){
        if(box.first.right!=box.first.left && box.first.bottom!=box.first.top){
            int first_x = min(box.first.left, box.first.right);
            int second_x = max(box.first.left, box.first.right);
            int first_y = min(box.first.top, box.first.bottom);
            int second_y = max(box.first.top, box.first.bottom);
            file_content += std::to_string(first_x) + " " + std::to_string(second_x) + " " + std::to_string(first_y) + " " + std::to_string(second_y) + " " + std::to_string((int)box.second.x) + " " + std::to_string((int)box.second.y) + "\n";
        }
    }

    // Write the string to a file
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

    // Clear the screen
    particles.clear();
    lines.clear();
    boxes.clear();
    InvalidateRect(m_hwnd, NULL, FALSE);
}

// Update the last known position of the mouse
void SimulationBuilder::updateMousePosition(int mouseX, int mouseY){
    this->mouseX = mouseX;
    this->mouseY = mouseY;
}