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

#define MOUSE_MOV 0
#define MOUSE_LEFT 1
#define MOUSE_RIGHT 2

class SimulationBuilder{
    private:
        HWND m_hwnd;

        std::vector<D2D1_ELLIPSE> particles;
        std::vector<std::pair<D2D1_POINT_2F, D2D1_POINT_2F>> lines;

        int mouseX = 0;
        int mouseY = 0;
        int current_state = 0;

        void addParticle(int x, int y);
        void doNothing(int x, int y);
        void startLine(int x, int y);
        void moveLine(int x, int y);

        // Transition table for the automaton defining the behavior for mouse events
        int transition_table[10][3] = {
            {0, 1, 8},
            {2, 0, 1},
            {3, 0, 2},
            {4, 0, 3},
            {5, 0, 4},
            {6, 0, 5},
            {7, 0, 6},
            {1, 0, 7},
            {9, 9, 0},
            {9, 9, 0},
        };

        // Output/action table for the automaton defining the behavior for mouse events
        void (SimulationBuilder::*action_table[10])(int, int) = {
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::doNothing,
            &SimulationBuilder::addParticle,
            &SimulationBuilder::startLine,
            &SimulationBuilder::moveLine
        };

        void storeAndClear();

    public:
        SimulationBuilder(HWND m_hwnd){
            this->m_hwnd = m_hwnd;
        };
        
        void mouseEvent(int x, int y, int event_type);

        std::vector<D2D1_ELLIPSE>& getParticles();

        std::vector<std::pair<D2D1_POINT_2F, D2D1_POINT_2F>>& getLines();

        void keyboardEvent(short key);
};

#endif