#include "envDAG.hpp"
#include <armadillo>
#include <iostream>
 
int main()
{
  
  // example of testing for success
 std::string datafile = "n4m1000.csv";  


std::cout << "Hello World!" << std::endl;

envDAG env1(datafile, 6,6);

  auto myArea = env1.getAlpha_w();
  std::cout<<env1.getAlpha_w()<<std::endl;
  std::cout<<myArea<<std::endl;

envDAG env2(datafile, 31.0);

  auto myArea2 = env2.getAlpha_w();
  std::cout<<env2.getAlpha_w()<<std::endl;
  std::cout<<myArea2<<std::endl;  

  env1.PrintData();

//D.save("test.txt",arma::arma_ascii);


arma::mat A = arma::randu<arma::mat>(2,2);
arma::mat B = arma::randu<arma::mat>(2,2);

arma::mat K = arma::kron(A,B);

arma::cout<<K;


  return 0;
}
