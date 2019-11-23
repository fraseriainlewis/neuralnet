#include "envDAG.hpp"
#include <armadillo>
#include <iostream>
 
int main()
{
  
  // example of testing for success
 std::string datafile = "n4m1000.csv";// "test3.csv"; 


std::cout << "Hello World!" << std::endl;

envDAG env1(datafile, 30,30);

 /* auto myArea = env1.getAlpha_w();
  std::cout<<env1.getAlpha_w()<<std::endl;
  std::cout<<myArea<<std::endl;
*/

 // env1.PrintData();
  env1.fitDAG();


  return 0;
}
