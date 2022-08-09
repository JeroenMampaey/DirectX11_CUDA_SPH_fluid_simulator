#ifndef SIMULATION_BUILDER_H
#define SIMULATION_BUILDER_H

#include <vector>
#include <d2d1.h>
#include <utility>
#include <string>

#define CEILING 0
#define FLOOR 700
#define RIGHT 1280
#define LEFT 0

#define RADIUS 7.5
#define MOVEMENTS_PER_PARTICLE 8

#define MOUSE_CLICK 0
#define MOUSE_MOVE 1
#define W_CLICK 2
#define B_CLICK 3
#define P_CLICK 4
#define LEFT_KEY_CLICK 5
#define RIGHT_KEY_CLICK 6
#define UP_KEY_CLICK 7
#define DOWN_KEY_CLICK 8
#define A_CLICK 9

#define DEFAULT_PUMP_VELOCITY 200

class SimulationBuilder{
    private:
        HWND m_hwnd;

        std::vector<D2D1_ELLIPSE> particles;
        std::vector<std::pair<D2D1_POINT_2F, D2D1_POINT_2F>> lines;
        std::vector<std::pair<D2D1_RECT_F, D2D1_POINT_2F>> boxes;

        int mouseX = 0;
        int mouseY = 0;
        int current_state = 0;

        void addParticle();
        void doNothing();
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
        int transition_table[19][10] = {
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
        void (SimulationBuilder::*action_table[19])() = {
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::addParticle,

            &SimulationBuilder::doNothing,
            &SimulationBuilder::startLine,
            &SimulationBuilder::moveLine,

            &SimulationBuilder::doNothing,
            &SimulationBuilder::startBox,
            &SimulationBuilder::moveBox,
            &SimulationBuilder::moveBoxDirectionLeft,
            &SimulationBuilder::moveBoxDirectionRight,
            &SimulationBuilder::moveBoxVelocityUp,
            &SimulationBuilder::moveBoxVelocityDown,

            &SimulationBuilder::store,
        };

    public:
        SimulationBuilder(HWND m_hwnd){
            this->m_hwnd = m_hwnd;
        };
        
        void event(int event_type);

        std::vector<D2D1_ELLIPSE>& getParticles();

        std::vector<std::pair<D2D1_POINT_2F, D2D1_POINT_2F>>& getLines();

        std::vector<std::pair<D2D1_RECT_F, D2D1_POINT_2F>>& getBoxes();

        void updateMousePosition(int mouseX, int mouseY);
};

#endif