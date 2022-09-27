#include "entity_manager.h"

// 65535 lines for particles: max 4+3 characters plus whitespace and '\n' -> 589815
// 100 lines for boundaries: max 4+3+4+3 characters plus 3 whitespaces and '\n' -> 1800
// 15 lines for pumps: max 4+3+4+3+4+4 characters plus 5 whitespaces and '\n' -> 420
#define MAX_BUFFERSIZE 592035

#define DEFAULT_NUMPOINTS 1500
#define DEFAULT_NUMBOUNDARIES 3

Point::Point(float x, float y) noexcept
    :
    x(x),
    y(y)
{}

Vector::Vector(float x, float y) noexcept
    :
    x(x),
    y(y)
{}

Boundary::Boundary(float x1, float y1, float x2, float y2)
    :
    point1(Point(x1, y1)),
    point2(Point(x2, y2)),
    lengthSquared((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)),
    normal(Vector((y2-y1)/sqrt(lengthSquared), (x1-x2)/sqrt(lengthSquared))),
    direction(Vector((x2-x1)/sqrt(lengthSquared), (y2-y1)/sqrt(lengthSquared)))
{
    if(x1==x2 && y1==y2){
        throw std::exception("A boundary with no length exists which is not allowed");
    }
}

Neighbour::Neighbour(const Particle& p, float kernel, float gradKernel) noexcept
    :
    p(p),
    kernel(kernel),
    gradKernel(gradKernel)
{}

VirtualNeighbour::VirtualNeighbour(const Particle& p, float kernel, float gradKernel, float virtualX, float virtualY) noexcept
    :
    Neighbour(p, kernel, gradKernel),
    virtualX(virtualX),
    virtualY(virtualY)
{}

Particle::Particle(float x, float y) noexcept
    :
    pos(Point(x, y)),
    oldPos(Point(x, y)),
    vel(Vector(0.0f, 0.0f))
{}

Pump::Pump(float xLow, float xHigh, float yLow, float yHigh, float velocityX, float velocityY) noexcept
    :
    leftBottom(Point(xLow, xLow)),
    rightTop(Point(xHigh, yHigh)),
    vel(Vector(velocityX, velocityY))
{}

EntityManager::EntityManager(){
    HANDLE hFile;
    char   ReadBuffer[MAX_BUFFERSIZE] = {0};

    hFile = CreateFileA(SLD_PATH_CONCATINATED("simulation2D.txt"), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

    if(hFile == INVALID_HANDLE_VALUE){
        buildDefaultSimulationLayout();
        return;
    }

    DWORD  dwBytesRead = 0;

    if(FALSE == ReadFile(hFile, ReadBuffer, MAX_BUFFERSIZE-1, static_cast<LPDWORD>(&dwBytesRead), NULL)){
        CloseHandle(hFile);
        buildDefaultSimulationLayout();
        return;
    }

    if (dwBytesRead > 0 && dwBytesRead <= MAX_BUFFERSIZE-1){
        ReadBuffer[dwBytesRead]='\0';
        buildSimulationLayoutFromFile(ReadBuffer);
    }
    else{
        buildDefaultSimulationLayout();
    }

    CloseHandle(hFile);
}

void EntityManager::buildDefaultSimulationLayout(){
    boundaries.push_back(Boundary(0.0f, HEIGHT, 0.0f, 0.0f));
    boundaries.push_back(Boundary(0.0f, 0.0f, WIDTH, 0.0f));
    boundaries.push_back(Boundary(WIDTH, 0.0f, WIDTH, HEIGHT));

    float start_x = 2*RADIUS;
    float end_x = WIDTH/4.0f;
    float start_y = 2*RADIUS;
    float end_y = 3.0f*HEIGHT/4.0f;
    float interval = sqrt((end_x-start_x)*(end_y-start_y)/DEFAULT_NUMPOINTS);
    float x = start_x;
    float y = start_y;
    for(int i=0; i<DEFAULT_NUMPOINTS; i++){
        particles.push_back(Particle(x, y));
        y = (x+interval > end_x) ? y+interval : y;
        x = (x+interval > end_x) ? start_x : x+interval;
    }
}

//TODO: make this parser more error prone (a wrongfully formatted file can easily crash the program at the moment)
void EntityManager::buildSimulationLayoutFromFile(char* buffer){
    // Skip the first line which should contain "PARTICLES:\n"
    int index = 10;
    if(buffer[index]=='\r') index++;
    index++;
    
    // Parse all particles until "LINES:\n" is found
    while(buffer[index]!='L'){
        // Read the x coordinate of the particle
        int first_number = 0;
        while(buffer[index]!=' '){
            first_number = first_number*10 + (buffer[index]-'0');
            index++;
        }
        index++;
        // Read the y coordinate of the particle
        int second_number = 0;
        while(buffer[index]!='\n' && buffer[index]!='\r'){
            second_number = second_number*10 + (buffer[index]-'0');
            index++;
        }
        if(buffer[index]=='\r') index++;
        index++;
        particles.push_back(Particle(first_number, second_number));
    }

    // Skip "LINES:\n"
    index += 6;
    if(buffer[index]=='\r') index++;
    index++;

    // Parse all boundaries until "PUMPS:\n" is found
    while(buffer[index]!='P'){
        // Read the x coordinate of the first point of the line
        int first_number = 0;
        while(buffer[index]!=' '){
            first_number = first_number*10 + (buffer[index]-'0');
            index++;
        }
        index++;
        // Read the y coordinate of the first point of the line
        int second_number = 0;
        while(buffer[index]!=' '){
            second_number = second_number*10 + (buffer[index]-'0');
            index++;
        }
        index++;
        // Read the x coordinate of the second point of the line
        int third_number = 0;
        while(buffer[index]!=' '){
            third_number = third_number*10 + (buffer[index]-'0');
            index++;
        }
        index++;
        // Read the y coordinate of the second point of the line
        int fourth_number = 0;
        while(buffer[index]!='\n' && buffer[index]!='\r'){
            fourth_number = fourth_number*10 + (buffer[index]-'0');
            index++;
        }
        if(buffer[index]=='\r') index++;
        index++;
        boundaries.push_back(Boundary(first_number, second_number, third_number, fourth_number));
    }

    // Skip "PUMPS:\n"
    index += 6;
    if(buffer[index]=='\r') index++;
    index++;

    // Parse all boundaries until the end of the array (marked by '\0')
    while(buffer[index]!='\0'){
        // Read the lowest x coordinate of the pump rectangle
        int first_number = 0;
        while(buffer[index]!=' '){
            first_number = first_number*10 + (buffer[index]-'0');
            index++;
        }
        index++;
        // Read the highest x coordinate of the pump rectangle
        int second_number = 0;
        while(buffer[index]!=' '){
            second_number = second_number*10 + (buffer[index]-'0');
            index++;
        }
        index++;
        // Read the lowest y coordinate of the pump rectangle
        int third_number = 0;
        while(buffer[index]!=' '){
            third_number = third_number*10 + (buffer[index]-'0');
            index++;
        }
        index++;
        // Read the highest y coordinate of the pump rectangle
        int fourth_number = 0;
        while(buffer[index]!=' '){
            fourth_number = fourth_number*10 + (buffer[index]-'0');
            index++;
        }
        index++;
        // Read the x coordinate of the velocity vector of the pump
        int fifth_number = 0;
        int sign = 1;
        if(buffer[index]=='-'){
            sign = -1;
            index++;
        }
        while(buffer[index]!=' '){
            fifth_number = fifth_number*10 + (buffer[index]-'0');
            index++;
        }
        fifth_number *= sign;
        index++;
        // Read the y coordinate of the velocity vector of the pump
        int sixth_number = 0;
        sign = 1;
        if(buffer[index]=='-'){
            sign = -1;
            index++;
        }
        while(buffer[index]!='\n' && buffer[index]!='\r'){
            sixth_number = sixth_number*10 + (buffer[index]-'0');
            index++;
        }
        sixth_number *= sign;
        if(buffer[index]=='\r') index++;
        index++;
        pumps.push_back(Pump(first_number, second_number, third_number, fourth_number, fifth_number, sixth_number));
    }
}

std::vector<Particle>& EntityManager::getParticles() noexcept{
    return particles;
}

std::vector<Boundary>& EntityManager::getBoundaries() noexcept{
    return boundaries;
}

std::vector<Pump>& EntityManager::getPumps() noexcept{
    return pumps;
}