#include "envDAG.hpp"
#include <armadillo>
#include <iostream>
 
int main()
{
  

/* basic operations are: 
   1. setup env - pre-compute as much as possible
   2. reset env to given state - this is a dag and position on board
   3. take an action and get the reward/updated state

*/
  // example of testing for success
 std::string datafile = "n10m1000a.csv";// "test3.csv"; 


envDAG env1(datafile);// assumes priors=30|30, and empty dag alpha_w, alpha_m


//env1.fitDAG();

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

arma::ivec posa = {2,1};// (x,y)

env1.resetDAG(daga,posa);
env1.step(0);// no move add arc at 2,1
env1.step(3);// left move and add arc at 2,0
env1.step(7);// right move only 2,1
env1.step(7);// right move only 2,2
env1.step(7);// right move only 2,3
env1.step(0);// add arc at 2,3
env1.step(2);// remove arc at 2,3

/* env1.step(6);// right 
env1.step(6);// right 
env1.step(9);// up
env1.step(6);// right 
env1.step(6);// right
env1.step(13);// down
env1.step(13);// down
*/
/*if(env1.hasCycle(daga)){std::cout<<"CYCLE!!"<<std::endl;
} else {env1.fitDAG();}


// add cycle to check it croaks
daga(1,8)=1;
env1.resetDAG(daga,posa);
if(env1.hasCycle(daga)){std::cout<<"CYCLE!!"<<std::endl;
} else {env1.fitDAG();}

std::string datafile2 = "n10m1000b.csv";// "test3.csv"; 

envDAG env2(datafile2,25,25);// assumes priors=25,25, and empty dag alpha_w, alpha_m
env2.fitDAG();
*/
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

arma::ivec posb = {9,2};

env1.resetDAG(dagb,posb);
if(env1.hasCycle()){std::cout<<"CYCLE!!"<<std::endl;
} else {env1.fitDAG();}

 
// add cycle to check it croaks
/*dagb(3,9)=1;
env2.resetDAG(dagb,posb);
if(env2.hasCycle(dagb)){std::cout<<"CYCLE!!"<<std::endl;
} else {env2.fitDAG();}
*/




  return 0;
}
