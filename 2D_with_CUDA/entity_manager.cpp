#include "entity_manager.h"

// 65535 lines for particles: max 4+3 characters plus whitespace and '\n' -> 589815
// 100 lines for boundaries: max 4+3+4+3 characters plus 3 whitespaces and '\n' -> 1800
// 15 lines for pumps: max 4+3+4+3+4+4 characters plus 5 whitespaces and '\n' -> 420
#define MAX_BUFFERSIZE 592035

#define DEFAULT_NUMPOINTS 1500
#define DEFAULT_NUMBOUNDARIES 3

EntityManager::EntityManager(GraphicsEngine& gfx){
    HANDLE hFile;
    char   ReadBuffer[MAX_BUFFERSIZE] = {0};

    hFile = CreateFileA(SLD_PATH_CONCATINATED("simulation2D.txt"), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);

    if(hFile == INVALID_HANDLE_VALUE){
        buildDefaultSimulationLayout(gfx);
        return;
    }

    DWORD  dwBytesRead = 0;

    if(FALSE == ReadFile(hFile, ReadBuffer, MAX_BUFFERSIZE-1, static_cast<LPDWORD>(&dwBytesRead), NULL)){
        CloseHandle(hFile);
        buildDefaultSimulationLayout(gfx);
        return;
    }

    if (dwBytesRead > 0 && dwBytesRead <= MAX_BUFFERSIZE-1){
        ReadBuffer[dwBytesRead]='\0';
        buildSimulationLayoutFromFile(gfx, ReadBuffer);
    }
    else{
        buildDefaultSimulationLayout(gfx);
    }

    CloseHandle(hFile);
}

void EntityManager::buildDefaultSimulationLayout(GraphicsEngine& gfx){
    std::vector<Particle> tempParticles;

    boundaries.push_back({0, HEIGHT, 0, 0});
    boundaries.push_back({0, 0, WIDTH, 0});
    boundaries.push_back({WIDTH, 0, WIDTH, HEIGHT});

    float start_x = 2*RADIUS;
    float end_x = WIDTH/4.0f;
    float start_y = 2*RADIUS;
    float end_y = 3.0f*HEIGHT/4.0f;
    float interval = sqrt((end_x-start_x)*(end_y-start_y)/DEFAULT_NUMPOINTS);
    float x = start_x;
    float y = start_y;
    for(int i=0; i<DEFAULT_NUMPOINTS; i++){
        tempParticles.push_back({x, y, 0.0f, 0.0f});
        y = (x+interval > end_x) ? y+interval : y;
        x = (x+interval > end_x) ? start_x : x+interval;
    }

    if(tempParticles.size()>MAX_POSSIBLE_PARTICLES){
        throw std::exception(("2D_with_CUDA simulations can support only up to "+std::to_string(MAX_POSSIBLE_PARTICLES)+" particles, not more.").c_str());
    }

    pParticles = gfx.createNewGraphicsBoundObject<CudaAccessibleFilledCircleInstanceBuffer>(static_cast<int>(tempParticles.size()), RADIUS);

    Particle* realParticles = (Particle*)pParticles->getMappedAccess();
    cudaError_t err;
    CUDA_THROW_FAILED(cudaMemcpy(realParticles, tempParticles.data(), sizeof(Particle)*tempParticles.size(), cudaMemcpyHostToDevice));
    pParticles->unMap();
}

//TODO: make this parser more error prone (a wrongfully formatted file can easily crash the program at the moment)
void EntityManager::buildSimulationLayoutFromFile(GraphicsEngine& gfx, char* buffer){
    std::vector<Particle> tempParticles;

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
        tempParticles.push_back({(float)first_number, (float)second_number, 0.0f, 0.0f});
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
        boundaries.push_back({(unsigned short)first_number, (unsigned short)second_number, (unsigned short)third_number, (unsigned short)fourth_number});
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
        pumps.push_back({(unsigned short)first_number, (unsigned short)second_number, (unsigned short)third_number, (unsigned short)fourth_number});
        pumpVelocities.push_back({(short)(((float)fifth_number)/100.0f), (short)(((float)sixth_number)/100.0f)});
    }

    if(tempParticles.size()>MAX_POSSIBLE_PARTICLES){
        throw std::exception(("2D_with_CUDA simulations can support only up to "+std::to_string(MAX_POSSIBLE_PARTICLES)+" particles, not more.").c_str());
    }

    pParticles = gfx.createNewGraphicsBoundObject<CudaAccessibleFilledCircleInstanceBuffer>(static_cast<int>(tempParticles.size()), RADIUS);

    Particle* realParticles = (Particle*)pParticles->getMappedAccess();
    cudaError_t err;
    CUDA_THROW_FAILED(cudaMemcpy(realParticles, tempParticles.data(), sizeof(Particle)*tempParticles.size(), cudaMemcpyHostToDevice));
    pParticles->unMap();
}

CudaAccessibleFilledCircleInstanceBuffer& EntityManager::getParticles() noexcept{
    return *pParticles.get();
}

std::vector<Boundary>& EntityManager::getBoundaries() noexcept{
    return boundaries;
}

std::vector<Pump>& EntityManager::getPumps() noexcept{
    return pumps;
}

std::vector<PumpVelocity>& EntityManager::getPumpVelocities() noexcept{
    return pumpVelocities;
}