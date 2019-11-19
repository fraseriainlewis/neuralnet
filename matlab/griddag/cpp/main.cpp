#include "envDAG.hpp"

#include <iostream>
 
int main()
{
  std::cout << "Hello World!" << std::endl;
  envDAG env1(10.0,20.0);

  auto myArea = env1.Area();
  std::cout<<env1.Area()<<std::endl;
  std::cout<<myArea<<std::endl;


  return 0;
}
