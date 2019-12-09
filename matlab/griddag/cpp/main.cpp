#include "envDAG.hpp"
#include <armadillo>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
 

#include <iostream>
#include <unordered_map>

#define Aa

template<typename K, typename V>
void print_map(std::unordered_map<K,V> const &m)
{
    for (auto const& pair: m) {
        std::cout << "{" << pair.first << ": " << pair.second << "}\n";
    }
}



int main()
{
  
unsigned int i;
double best_value;
arma::umat curDAG;
arma::ivec curPos;
std::string curDagKey;
/* basic operations are: 
   1. setup env - pre-compute as much as possible
   2. reset env to given state - this is a dag and position on board
   3. take an action and get the reward/updated state

*/
// set file with observed data
std::string datafile = "n4m1000.csv";// "test3.csv"; 

// set up random number generator - for breaking ties and random starts
long unsigned int seed=1000;
std::mt19937 engine(seed);  // Mersenne twister random number engine
std::uniform_real_distribution<double> distr(0.0, 1.0);//call using distr(engine) to get U(0,1) variate

//std::cout<<"random U(0,1)="<<distr(engine)<<" and another "<<distr(engine)<<std::endl;

double discount = 0.9;// discounting for value function
double curQ;
unsigned int greedyA;

// set up environment which does:
// reads in data and computes necessary constants for reward (ln network score) which are independent of DAG structure so can be pre-computed
// based on observed data and prior. Default prior is alpha_w,alpha_m = 30|30. This also sets the dag, dag0 to empty dag and the position on the
// gridworld is 0,0 top left cornder and creates a unique key for this state.
envDAG env1(datafile, -6470.0);


std::cout<<"initial reward="<<env1.fitDAG()<<std::endl;

#ifdef A
        0        0        1        1        0        0        0        0        0        0
        1        0        0        1        0        0        0        0        0        0
        0        0        0        1        0        0        0        0        0        0
        0        0        0        0        0        0        0        0        0        0
        0        0        0        0        0        0        0        0        0        0
        0        0        0        0        0        0        0        1        0        0
        1        0        0        0        1        1        0        1        0        0
        0        0        0        0        0        0        0        0        0        0
        0        0        0        0        0        0        0        0        0        0
        1        0        0        0        0        0        0        0        1        0

#endif
#ifdef B
arma::umat daga = { 
      {0,    0,    1,    1,    0,    0,    0,    0,    0,     0},
      {1,    0,    0,    1,    0,    0,    0,    0,    0,     0},
      {0,    0,    0,    1,    0,    0,    0,    0,    0,     0},
      {0,    0,    0,    0,    0,    0,    0,    0,    0,     0},
      {0,    0,    0,    0,    0,    0,    0,    0,    0,     0},
      {0,    0,    0,    0,    0,    0,    0,    1,    0,     0},
      {1,    0,    0,    0,    1,    1,    0,    1,    0,     0},
      {0,    0,    0,    0,    0,    0,    0,    0,    0,     0},
      {0,    0,    0,    0,    0,    0,    0,    0,    0,     0},
      {1,    0,    0,    0,    0,    0,    0,    0,    1,     0}
           };

arma::ivec posa = {0,0};// (x,y)

env1.resetDAG(daga,posa);
std::cout<<"my reward="<<env1.fitDAG()<<std::endl;
arma::cout<<env1.dag0<<arma::endl;
exit(1);
#endif

curDAG=env1.dag0; // copy - this is not efficient - fix later as no need to recreate memory
curPos=env1.pos0; // copy - this is not efficient - fix later as no need to recreate memory
curDagKey=env1.dagkey;// copy current dagkey

arma::cout<<"First State="<<arma::endl<<env1.dagkey<<arma::endl;
//exit(1);

//arma::cout<<"dag0 pos0"<<arma::endl<<curDAG<<arma::endl<<curPos<<arma::endl;

// from initial state run through all actions, take the best and then update the value function
// 1. store current DAG

unsigned int steps;

// we start with an empty dag - constructor sets IsDone to false

double bestscore=-std::numeric_limits<double>::max();
arma::umat bestdag;
unsigned int period;

arma::umat dagnull=arma::zeros<arma::umat>(env1.n,env1.n);
arma::ivec posnull = {0,0};// (x,y)

unsigned int numPeriods=1000;

arma::uvec stepcount(numPeriods);

for(period=0;period<numPeriods;period++){
std::cout<<"PERIOD="<<period<<std::endl;
curDAG=dagnull;
curPos=posnull;

env1.resetDAG(curDAG,curPos);// reset start of period to null model
//std::cout<<"initial reward="<<env1.fitDAG()<<"->"<<env1.reward<<"->"<<env1.IsDone<<std::endl;
curDagKey=env1.dagkey;// copy current dagkey

steps=1;
while(!env1.IsDone && steps<=250)
{

// loop through each actions and take best
best_value=-std::numeric_limits<double>::max();//worst possible reward

for(i=0;i<15;i++){
  env1.resetDAG(curDAG,curPos);//reset back to current state including dagkey
  env1.step(i);// take action i 

  if(env1.invalidAction){continue;} // got a bad action, e.g. cycle created so skip to next iteration
    // have a valid action so update values
          
          curQ = env1.reward + discount*env1.getValue(env1.dagkey); // dagkey is for NEW state
          //std::cout<<"reward="<<env1.reward<<" i="<<i<<std::endl;
          if (curQ > best_value){// found a better action
              best_value = curQ;
              /*if (env1.ValueMap.find(curDagKey) != env1.ValueMap.end()){std::cout<<"UPDATING EXISTING VALUE!"<<std::endl;
                                                                        std::cout<<"old value="<<env1.ValueMap[curDagKey]<<std::endl;} */

              env1.ValueMap[curDagKey] = curQ; // update value function for just this CURRENT state
              //std::cout<<"new value="<<env1.ValueMap[curDagKey]<<std::endl;
              greedyA = i; // store best action

         } else {
                  if(curQ==best_value && distr(engine)>0.5){// action is same as current best action so a tie, break randomly if U(0,1)>0.5
                      best_value = curQ;
                      //if (env1.ValueMap.find(curDagKey) != env1.ValueMap.end()){std::cout<<"UPDATING EXISTING VALUE!"<<std::endl;} 
                      env1.ValueMap[curDagKey] = curQ; // update value function for just this CURRENT state
                      greedyA = i; // store best action

                  } 
                }

}

//std::cout<<"best action="<<greedyA<<std::endl<<" reward="<<env1.reward<<std::endl;
env1.resetDAG(curDAG,curPos);// resets IsDone to false
env1.step(greedyA);// take best action i and update current state to this - this might set IsDone to true and terminate episode

if(env1.fitDAG()>bestscore){bestscore=env1.fitDAG();bestdag=env1.dag0;}

//std::cout<<"current reward="<<env1.fitDAG()<<std::endl;
// now copy the current state and repeat action search
curDAG=env1.dag0; // copy - this is not efficient - fix later as no need to recreate memory
curPos=env1.pos0; // copy - this is not efficient - fix later as no need to recreate memory
curDagKey=env1.dagkey;// copy current dagkey


steps++;
} // end of episode loop/while
stepcount(period)=steps-1.0;


} // end of period loop
arma::cout<<"stepcounts"<<arma::endl<<stepcount<<arma::endl;

#ifdef Aa
arma::cout<<"final state="<<arma::endl<<env1.dagkey<<arma::endl;
std::cout<<"final reward="<<env1.fitDAG()<<std::endl;

std::cout<<"best score visited="<<bestscore<<std::endl;
arma::cout<<"best DAG visited="<<bestdag<<arma::endl;

//print_map(env1.ValueMap);
std::cout<<"number of states stored="<<env1.ValueMap.size()<<std::endl;
#endif


/*
std::ostringstream s;
for (auto const& pair: env1.ValueMap) {
        s << pair.first << "," << pair.second << std::endl;
    }

std::cout<<std::endl<<s.str();

    std::ofstream out("output.txt");
    out << s.str();
    out.close();
*/



  return 0;
}
