#pragma once

#include "../lib/lib_header.h"
#include <vector>
#include <utility>
#include <cmath>
#include "exceptions.h"

#pragma comment(lib,"xmllite.lib")
#pragma comment(lib,"Shlwapi.lib")
#include "xmllite.h"
#include "Shlwapi.h"
#include <tchar.h>
#include <wchar.h>
#include <stdexcept>

#ifndef SIMULATION_LAYOUT_DIRECTORY
#define SIMULATION_LAYOUT_DIRECTORY "../../simulation_layout/"
#endif

#define SLD_PATH_CONCATINATED(original) SIMULATION_LAYOUT_DIRECTORY original

#define RADIUS 2.0f
#define PI 3.141592
#define SQRT_PI 1.772453

#define Particle DirectX::XMFLOAT4

struct Boundary{
    unsigned short x1;
    unsigned short y1;
    unsigned short x2;
    unsigned short y2;
};

struct Pump{
    unsigned short xLow;
    unsigned short xHigh;
    unsigned short yLow;
    unsigned short yHigh;
};

struct PumpVelocity{
    float velX;
    float velY;
};

class EntityManager{
    public:
        EntityManager(GraphicsEngine& gfx);

        CudaAccessibleFilledCircleInstanceBuffer& getParticles() noexcept;
        std::vector<Boundary>& getBoundaries() noexcept;
        std::vector<Pump>& getPumps() noexcept;
        std::vector<PumpVelocity>& getPumpVelocities() noexcept;
    
    private:
        void buildDefaultSimulationLayout(GraphicsEngine& gfx);
        bool readAttributeDouble(IXmlReader* pReader, const WCHAR* attributeName, double* readDouble) noexcept;
        bool readAttributeInteger(IXmlReader* pReader, const WCHAR* attributeName, int* readInt) noexcept;

        std::vector<Boundary> boundaries;
        std::vector<Pump> pumps;
        std::unique_ptr<CudaAccessibleFilledCircleInstanceBuffer> pParticles;
        std::vector<PumpVelocity> pumpVelocities;
};