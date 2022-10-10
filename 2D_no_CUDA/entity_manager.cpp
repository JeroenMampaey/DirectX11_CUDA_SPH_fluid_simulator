#include "entity_manager.h"


// 65535 lines for particles: max 4+3 characters plus whitespace and '\n' -> 589815
// 100 lines for boundaries: max 4+3+4+3 characters plus 3 whitespaces and '\n' -> 1800
// 15 lines for pumps: max 4+3+4+3+4+4 characters plus 5 whitespaces and '\n' -> 420
#define MAX_BUFFERSIZE 592035

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
    leftBottom(Point(xLow, yLow)),
    rightTop(Point(xHigh, yHigh)),
    vel(Vector(velocityX, velocityY))
{}

EntityManager::EntityManager(){
    Microsoft::WRL::ComPtr<IStream> pInFileStream;
    Microsoft::WRL::ComPtr<IXmlReader> pReader;

    HRESULT hr;
    if(FAILED(hr = SHCreateStreamOnFileA(SLD_PATH_CONCATINATED("simulation2D.txt"), STGM_READ, &pInFileStream))){
        if(HRESULT_CODE(hr)==ERROR_FILE_NOT_FOUND){
            buildDefaultSimulationLayout();
            return;
        }
        else{
            throw std::exception("Found a simulation2D.txt file but could not create an reader stream for it.");
        }
    }

    if(FAILED(CreateXmlReader(__uuidof(IXmlReader), (void**) &pReader, NULL))){
        throw std::exception("Failed to create an XML reader for the simulation2D.txt file");
    }

    if(FAILED(pReader->SetProperty(XmlReaderProperty_DtdProcessing, DtdProcessing_Prohibit))){
        throw std::exception("Failed to set the dtdprocessing property for reading the simulation2D.txt file");
    }

    if(FAILED(pReader->SetInput(pInFileStream.Get()))){
        throw std::exception("Could not bind the XML reader for the simulation2D.txt file");
    }

    XmlNodeType nodeType;
    const WCHAR* localName;
    const WCHAR* value;
    UINT valueSize;

    double particleZoneRadius;
    std::map<float, std::vector<float>> particleZones;

    while(true){
        hr = pReader->Read(&nodeType);
        if(hr!=E_PENDING && FAILED(hr)){
            throw std::exception("Failed to read the simulation2D.txt file");
        }

        if(nodeType==XmlNodeType_XmlDeclaration){
            break;
        }
        else if(nodeType!=XmlNodeType_Whitespace){
            throw std::exception("Simulation2D.txt file did not start with an XmlDeclaration");
        }
    }

    while(true){
        hr = pReader->Read(&nodeType);
        if(hr!=E_PENDING && FAILED(hr)){
            throw std::exception("Failed to read the simulation2D.txt file");
        }

        if(nodeType==XmlNodeType_Element){
            if(FAILED(pReader->GetLocalName(&localName, NULL))
                || wcscmp(localName, L"Layout")!=0
            ){
                throw std::exception("Something is wrong with the Layout element of the simulation2D.txt file");
            }

            if(!readAttributeDouble(pReader.Get(), L"particleZoneRadius", &particleZoneRadius)){
                throw std::exception("Something is wrong with the particleZoneRadius attribute of the Layout element in de simulation2D.txt file");
            }

            break;
        }
        else if(nodeType!=XmlNodeType_Whitespace){
            throw std::exception("Layout element in the simulation2D.txt file is not correctly placed");
        }
    }

    while(true){
        hr = pReader->Read(&nodeType);
        if(hr!=E_PENDING && FAILED(hr)){
            throw std::exception("Failed to read the simulation2D.txt file");
        }

        if(nodeType==XmlNodeType_EndElement){
            if(FAILED(pReader->GetLocalName(&localName, NULL))
                || wcscmp(localName, L"Layout")!=0
            ){
                throw std::exception("Something is wrong with the Layout element of the simulation2D.txt file");
            }

            break;
        }
        else if(nodeType==XmlNodeType_Element){
            if(FAILED(pReader->GetLocalName(&localName, NULL))){
                throw std::exception("Something is wrong inside the Layout element of the simulation2D.txt file");
            }

            if(wcscmp(localName, L"ParticleZone")==0){
                int x;
                int y;
                if(!readAttributeInteger(pReader.Get(), L"x", &x)){
                    throw std::exception("Something is wrong with the x attribute of a ParticleZone element in de simulation2D.txt file");
                }
                if(!readAttributeInteger(pReader.Get(), L"y", &y)){
                    throw std::exception("Something is wrong with the y attribute of a ParticleZone element in de simulation2D.txt file");
                }
                particleZones[(float)y].push_back((float)x);
            }
            else if(wcscmp(localName, L"Boundary")==0){
                int x1;
                int y1;
                int x2;
                int y2;
                if(!readAttributeInteger(pReader.Get(), L"x1", &x1)){
                    throw std::exception("Something is wrong with the x1 attribute of a Boundary element in de simulation2D.txt file");
                }
                if(!readAttributeInteger(pReader.Get(), L"y1", &y1)){
                    throw std::exception("Something is wrong with the y1 attribute of a Boundary element in de simulation2D.txt file");
                }
                if(!readAttributeInteger(pReader.Get(), L"x2", &x2)){
                    throw std::exception("Something is wrong with the x2 attribute of a Boundary element in de simulation2D.txt file");
                }
                if(!readAttributeInteger(pReader.Get(), L"y2", &y2)){
                    throw std::exception("Something is wrong with the y2 attribute of a Boundary element in de simulation2D.txt file");
                }
                boundaries.push_back(Boundary(x1, y1, x2, y2));
            }
            else if(wcscmp(localName, L"Pump")==0){
                int xLow;
                int yLow;
                int xHigh;
                int yHigh;
                double velX;
                double velY;
                if(!readAttributeInteger(pReader.Get(), L"xLow", &xLow)){
                    throw std::exception("Something is wrong with the xLow attribute of a Pump element in de simulation2D.txt file");
                }
                if(!readAttributeInteger(pReader.Get(), L"yLow", &yLow)){
                    throw std::exception("Something is wrong with the yLow attribute of a Pump element in de simulation2D.txt file");
                }
                if(!readAttributeInteger(pReader.Get(), L"xHigh", &xHigh)){
                    throw std::exception("Something is wrong with the xHigh attribute of a Pump element in de simulation2D.txt file");
                }
                if(!readAttributeInteger(pReader.Get(), L"yHigh", &yHigh)){
                    throw std::exception("Something is wrong with the yHigh attribute of a Pump element in de simulation2D.txt file");
                }
                if(!readAttributeDouble(pReader.Get(), L"velX", &velX)){
                    throw std::exception("Something is wrong with the velX attribute of a Pump element in de simulation2D.txt file");
                }
                if(!readAttributeDouble(pReader.Get(), L"velY", &velY)){
                    throw std::exception("Something is wrong with the velY attribute of a Pump element in de simulation2D.txt file");
                }
                pumps.push_back(Pump(xLow, xHigh, yLow, yHigh, velX, velY));
            }
            else{
                throw std::exception("Unexpected element was found inside the Layout element of the simulation2D.txt file");
            }
        }
        else if(nodeType!=XmlNodeType_Whitespace){
            throw std::exception("Unexpected xml was found inside the Layout element of the simulation2D.txt file");
        }
    }

    while(!pReader->IsEOF()){
        hr = pReader->Read(&nodeType);
        if(hr!=E_PENDING && FAILED(hr)){
            throw std::exception("Failed to read the simulation2D.txt file");
        }

        if(nodeType!=XmlNodeType_Whitespace && nodeType!=XmlNodeType_None){
            throw std::exception("Simulation2D.txt file had some unexpected xml after the Layout element");
        }
    }
    
    if(particleZones.size()>0){
        for(float y=0.0f; y<HEIGHT; y+=SQRT_PI*RADIUS){
            auto lowerBound = particleZones.lower_bound(y);
            if((lowerBound==particleZones.end() || (lowerBound->first-y)*(lowerBound->first-y)>particleZoneRadius*particleZoneRadius)
                && (lowerBound==particleZones.begin() || (std::prev(lowerBound)->first-y)*(std::prev(lowerBound)->first-y)>particleZoneRadius*particleZoneRadius)
            ){
                continue;
            }
            for(float x=0.0f; x<WIDTH; x+=SQRT_PI*RADIUS){
                auto upper = lowerBound;
                auto lower = lowerBound;
                bool found = false;
                while(!found && upper!=particleZones.end() && (upper->first-y)*(upper->first-y)<=particleZoneRadius*particleZoneRadius){
                    for(auto it=upper->second.begin(); it!=upper->second.end(); it++){
                        if((*it-x)*(*it-x)+(upper->first-y)*(upper->first-y)<=particleZoneRadius*particleZoneRadius){
                            particles.push_back(Particle(x, y));
                            found = true;
                            break;
                        }
                    }
                    upper = std::next(upper);
                }
                while(!found && lower!=particleZones.begin() && (std::prev(lower)->first-y)*(std::prev(lower)->first-y)<=particleZoneRadius*particleZoneRadius){
                    lower = std::prev(lower);
                    for(auto it=lower->second.begin(); it!=lower->second.end(); it++){
                        if((*it-x)*(*it-x)+(lower->first-y)*(lower->first-y)<=particleZoneRadius*particleZoneRadius){
                            particles.push_back(Particle(x, y));
                            found = true;
                            break;
                        }
                    }
                }
            }
        }
    }
}

void EntityManager::buildDefaultSimulationLayout(){
    boundaries.push_back({0, HEIGHT, 0, 0});
    boundaries.push_back({0, 0, WIDTH, 0});
    boundaries.push_back({WIDTH, 0, WIDTH, HEIGHT});

    for(float y = RADIUS; y<0.5f*HEIGHT; y+=SQRT_PI*RADIUS){
        for(float x = RADIUS; x<0.33f*WIDTH; x+=SQRT_PI*RADIUS){
            particles.push_back(Particle(x, y));
        }
    }
}

bool EntityManager::readAttributeInteger(IXmlReader* pReader, const WCHAR* attributeName, int* readInt) noexcept{
    const WCHAR* localName;
    const WCHAR* value;
    UINT valueSize;

    if(FAILED(pReader->MoveToAttributeByName(attributeName, NULL))
        || FAILED(pReader->GetLocalName(&localName, NULL))
        || wcscmp(localName, attributeName)!=0
        || FAILED(pReader->GetValue(&value, &valueSize))
    ){
        return false;
    }
    
    std::wstring::size_type sz;
    try{
        *readInt = std::stoi(std::wstring(value), &sz);
    } catch(std::invalid_argument& e){
        return false;
    }

    if(sz!=valueSize){
        return false;
    }

    if(FAILED(pReader->MoveToElement())){
        return false;
    }

    return true;
}

bool EntityManager::readAttributeDouble(IXmlReader* pReader, const WCHAR* attributeName, double* readDouble) noexcept{
    const WCHAR* localName;
    const WCHAR* value;
    UINT valueSize;

    if(FAILED(pReader->MoveToAttributeByName(attributeName, NULL))
        || FAILED(pReader->GetLocalName(&localName, NULL))
        || wcscmp(localName, attributeName)!=0
        || FAILED(pReader->GetValue(&value, &valueSize))
    ){
        return false;
    }
    
    std::wstring::size_type sz;
    try{
        *readDouble = std::stod(std::wstring(value), &sz);
    } catch(std::invalid_argument& e){
        return false;
    }

    if(sz!=valueSize){
        return false;
    }

    if(FAILED(pReader->MoveToElement())){
        return false;
    }

    return true;
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