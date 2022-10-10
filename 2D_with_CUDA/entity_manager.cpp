#include "entity_manager.h"

// 65535 lines for particles: max 4+3 characters plus whitespace and '\n' -> 589815
// 100 lines for boundaries: max 4+3+4+3 characters plus 3 whitespaces and '\n' -> 1800
// 15 lines for pumps: max 4+3+4+3+4+4 characters plus 5 whitespaces and '\n' -> 420
#define MAX_BUFFERSIZE 592035

EntityManager::EntityManager(GraphicsEngine& gfx){
    Microsoft::WRL::ComPtr<IStream> pInFileStream;
    Microsoft::WRL::ComPtr<IXmlReader> pReader;

    HRESULT hr;
    if(FAILED(hr = SHCreateStreamOnFileA(SLD_PATH_CONCATINATED("simulation2D.txt"), STGM_READ, &pInFileStream))){
        if(HRESULT_CODE(hr)==ERROR_FILE_NOT_FOUND){
            buildDefaultSimulationLayout(gfx);
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
                if(x1>=0 && x1<=USHRT_MAX && y1>=0 && y1<=USHRT_MAX && x2>=0 && x2<=USHRT_MAX && y2>=0 && y2<=USHRT_MAX){
                    boundaries.push_back({(unsigned short)x1, (unsigned short)y1, (unsigned short)x2, (unsigned short)y2});
                }
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
                if(xLow>=0 && xLow<=USHRT_MAX && yLow>=0 && yLow<=USHRT_MAX && xHigh>=0 && xHigh<=USHRT_MAX && yHigh>=0 && yHigh<=USHRT_MAX){
                    pumps.push_back({(unsigned short)xLow, (unsigned short)xHigh, (unsigned short)yLow, (unsigned short)yHigh});
                    pumpVelocities.push_back({(float)velX, (float)velY});
                }
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

    std::vector<Particle> tempParticles;
    
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
                            tempParticles.push_back({x, y, 0.0f, 0.0f});
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
                            tempParticles.push_back({x, y, 0.0f, 0.0f});
                            found = true;
                            break;
                        }
                    }
                }
            }
        }
    }

    pParticles = gfx.createNewGraphicsBoundObject<CudaAccessibleFilledCircleInstanceBuffer>(static_cast<int>(tempParticles.size()), RADIUS);

    if(tempParticles.size()>0){
        Particle* realParticles = (Particle*)pParticles->getMappedAccess();
        cudaError_t err;
        CUDA_THROW_FAILED(cudaMemcpy(realParticles, tempParticles.data(), sizeof(Particle)*tempParticles.size(), cudaMemcpyHostToDevice));
        pParticles->unMap();
    }
}

void EntityManager::buildDefaultSimulationLayout(GraphicsEngine& gfx){
    std::vector<Particle> tempParticles;

    boundaries.push_back({0, HEIGHT, 0, 0});
    boundaries.push_back({0, 0, WIDTH, 0});
    boundaries.push_back({WIDTH, 0, WIDTH, HEIGHT});

    for(float y = RADIUS; y<0.5f*HEIGHT; y+=SQRT_PI*RADIUS){
        for(float x = RADIUS; x<0.33f*WIDTH; x+=SQRT_PI*RADIUS){
            tempParticles.push_back({x, y, 0.0f, 0.0f});
        }
    }

    pParticles = gfx.createNewGraphicsBoundObject<CudaAccessibleFilledCircleInstanceBuffer>(static_cast<int>(tempParticles.size()), RADIUS);

    if(tempParticles.size()>0){
        Particle* realParticles = (Particle*)pParticles->getMappedAccess();
        cudaError_t err;
        CUDA_THROW_FAILED(cudaMemcpy(realParticles, tempParticles.data(), sizeof(Particle)*tempParticles.size(), cudaMemcpyHostToDevice));
        pParticles->unMap();
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