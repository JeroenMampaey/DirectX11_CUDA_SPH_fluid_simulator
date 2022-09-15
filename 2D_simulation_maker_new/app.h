#pragma once

#include "../lib/lib_header.h"

#define RADIUS 7.5f
#define DEFAULT_PUMP_VELOCITY 200.0f

class App : public EventListener{
    public:
        App(Window& wnd);
        void handleEvent(const Event& event);
    
    private:
        Window& wnd;
        std::vector<FilledCircle*> particles;
        std::vector<Line*> boundaries;
        std::vector<HollowRectangle*> pumps;
        std::vector<Line*> pumpDirections;

        int mouseX = 0;
        int mouseY = 0;
        int currentState = 0;

        void doNothing();
        void addParticle();
        void startLine();
        void moveLine();
        void startBox();
        void moveBox();
        void moveBoxDirectionLeft();
        void moveBoxDirectionRight();
        void moveBoxVelocityUp();
        void moveBoxVelocityDown();
        void store();

        // Transition table for the automaton defining the transition behaviour for events
        int transitionTable[19][10] = {
            {1, 0, 0, 8, 11, 0, 0, 0, 0, 18},
            {0, 2, 1, 8, 11, 1, 1, 1, 1, 18},
            {0, 3, 2, 8, 11, 2, 2, 2, 2, 18},
            {0, 4, 3, 8, 11, 3, 3, 3, 3, 18},
            {0, 5, 4, 8, 11, 4, 4, 4, 4, 18},
            {0, 6, 5, 8, 11, 5, 5, 5, 5, 18},
            {0, 7, 6, 8, 11, 6, 6, 6, 6, 18},
            {0, 1, 1, 8, 11, 1, 1, 1, 1, 18},

            {9, 8, 0, 8, 11, 8, 8, 8, 8, 18},
            {8, 10, 0, 10, 11, 9, 9, 9, 9, 18},
            {8, 10, 0, 10, 11, 10, 10, 10, 10, 18},

            {12, 11, 0, 8, 11, 11, 11, 11, 11, 18},
            {11, 13, 0, 8, 13, 14, 15, 16, 17, 18},
            {11, 13, 0, 8, 13, 14, 15, 16, 17, 18},
            {11, 13, 0, 8, 13, 14, 15, 16, 17, 18},
            {11, 13, 0, 8, 13, 14, 15, 16, 17, 18},
            {11, 13, 0, 8, 13, 14, 15, 16, 17, 18},
            {11, 13, 0, 8, 13, 14, 15, 16, 17, 18},

            {1, 0, 0, 8, 11, 0, 0, 0, 0, 18}
        };

        // Output/action table for the automaton defining the behaviour for events
        void (App::*actionTable[19])() = {
            &App::doNothing,
            &App::doNothing,
            &App::doNothing,
            &App::doNothing,
            &App::doNothing,
            &App::doNothing,
            &App::doNothing,
            &App::addParticle,

            &App::doNothing,
            &App::startLine,
            &App::moveLine,

            &App::doNothing,
            &App::startBox,
            &App::moveBox,
            &App::moveBoxDirectionLeft,
            &App::moveBoxDirectionRight,
            &App::moveBoxVelocityUp,
            &App::moveBoxVelocityDown,

            &App::store,
        };

        std::map<LPARAM, int> keyCodeHelper;
};