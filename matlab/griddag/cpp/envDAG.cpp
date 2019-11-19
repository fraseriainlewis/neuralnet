// envDAG.cpp
#include "envDAG.hpp"
#include <iostream>

envDAG::envDAG(const double _l, const double _w) : l(_l), w(_w) { // this is an initializer list - as an alternative, here it would be possible to just assign the values inside the function body
				std::cout<<"constructing!"<<std::endl;
}



double envDAG::Area(void) const {
    return l * w;
}

double envDAG::Perim(void) const {
    return l + l + w + w;
}
