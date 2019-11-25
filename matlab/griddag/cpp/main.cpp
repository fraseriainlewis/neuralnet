#include "envDAG.hpp"
#include <armadillo>
#include <iostream>
 
int main()
{
  
  // example of testing for success
 std::string datafile = "n10m1000a.csv";// "test3.csv"; 


envDAG env1(datafile);// assumes priors=30|30, and empty dag alpha_w, alpha_m
env1.fitDAG();

#ifdef A
       [,0] [,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9]
 [0,]    0    1    1    0    0    0    0    0    0     0
 [1,]    0    0    1    1    0    0    0    0    0     0
 [2,]    0    0    0    1    0    0    0    0    0     0
 [3,]    0    0    0    0    0    0    0    0    0     0
 [4,]    1    0    0    0    0    0    1    0    0     0
 [5,]    0    0    0    0    0    0    0    1    0     0
 [6,]    0    0    0    0    0    0    0    1    0     0
 [7,]    1    0    0    0    0    0    0    0    0     0
 [8,]    0    0    0    0    0    0    0    0    0     1
 [9,]    0    0    0    0    1    1    0    0    0     0
#endif

arma::umat daga = { 
    	{0,    1,    1,    0,    0,    0,    0,    0,    0,     0},
    	{0,    0,    1,    1,    0,    0,    0,    0,    0,     0},
     	{0,    0,    0,    1,    0,    0,    0,    0,    0,     0},
     	{0,    0,    0,    0,    0,    0,    0,    0,    0,     0},
     	{1,    0,    0,    0,    0,    0,    1,    0,    0,     0},
    	{0,    0,    0,    0,    0,    0,    0,    1,    0,     0},
    	{0,    0,    0,    0,    0,    0,    0,    1,    0,     0},
     	{1,    0,    0,    0,    0,    0,    0,    0,    0,     0},
    	{0,    0,    0,    0,    0,    0,    0,    0,    0,     1},
     	{0,    0,    0,    0,    1,    1,    0,    0,    0,     0}
           };

env1.resetDAG(daga);
if(env1.hasCycle(daga)){std::cout<<"CYCLE!!"<<std::endl;
} else {env1.fitDAG();}


// add cycle to check it croaks
daga(1,8)=1;
env1.resetDAG(daga);
if(env1.hasCycle(daga)){std::cout<<"CYCLE!!"<<std::endl;
} else {env1.fitDAG();}

std::string datafile2 = "n10m1000b.csv";// "test3.csv"; 

envDAG env2(datafile2,25,25);// assumes priors=25,25, and empty dag alpha_w, alpha_m
env2.fitDAG();

#ifdef A
[,1] [,2] [,3] [,4] [,5] [,6] [,7] [,8] [,9] [,10]
 [1,]    0    1    1    0    0    0    0    0    0     0
 [2,]    0    0    1    1    0    0    0    0    0     0
 [3,]    0    0    0    1    0    0    0    0    0     0
 [4,]    0    0    0    0    0    0    0    0    0     0
 [5,]    1    0    0    0    0    0    1    0    0     0
 [6,]    0    0    1    0    0    0    0    0    0     0
 [7,]    0    0    1    0    0    0    0    0    0     0
 [8,]    1    0    0    0    0    0    0    0    0     0
 [9,]    0    0    0    1    0    0    0    0    0     0
[10,]    0    1    0    0    0    0    0    0    0     0
#endif 

arma::umat dagb = { 
    	{0,    1,    1,    0,    0,    0,    0,    0,    0,     0},
    	{0,    0,    1,    1,    0,    0,    0,    0,    0,     0},
     	{0,    0,    0,    1,    0,    0,    0,    0,    0,     0},
     	{0,    0,    0,    0,    0,    0,    0,    0,    0,     0},
     	{1,    0,    0,    0,    0,    0,    1,    0,    0,     0},
    	{0,    0,    1,    0,    0,    0,    0,    0,    0,     0},
    	{0,    0,    1,    0,    0,    0,    0,    0,    0,     0},
     	{1,    0,    0,    0,    0,    0,    0,    0,    0,     0},
    	{0,    0,    0,    1,    0,    0,    0,    0,    0,     0},
     	{0,    1,    0,    0,    0,    0,    0,    0,    0,     0}
           };

env2.resetDAG(dagb);
if(env2.hasCycle(dagb)){std::cout<<"CYCLE!!"<<std::endl;
} else {env2.fitDAG();}

 
// add cycle to check it croaks
dagb(3,9)=1;
env2.resetDAG(dagb);
if(env2.hasCycle(dagb)){std::cout<<"CYCLE!!"<<std::endl;
} else {env2.fitDAG();}






  return 0;
}
