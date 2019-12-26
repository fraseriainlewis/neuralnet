#include "envDAG.hpp"
#include <armadillo>
#include <iostream>
#include <iomanip>
#include <random>
#include <string>
#include <unordered_map>
#include <ios>
#include <sstream>

#define Aa
#define globalcheck31 // needs Customprior1 set

template<typename K, typename V>
void print_map(std::unordered_map<K,V> const &m)
{
    for (auto const& pair: m) {
        std::cout << "{" << pair.first << ": " << pair.second << "}\n";
    }
}


int main()
{
  
std::ios oldState(nullptr);
oldState.copyfmt(std::cout);

unsigned int i;
double best_value,best_score_period;
arma::umat curDAG;
arma::ivec curPos;
std::string curDagKey;
std::string str;          // The string
std::stringstream temp;  // 'temp' as in temporary
/* basic operations are: 
   1. setup env - pre-compute as much as possible
   2. reset env to given state - this is a dag and position on board
   3. take an action and get the reward/updated state

*/
// set file with observed data
std::string datafile = "n20m10000b.csv";// "test3.csv"; //n20m10000a.csv  n10Chainm10000.csv

// set up random number generator - for breaking ties and random starts
long unsigned int seed=100001; // 100001 is good for N20 option a, 100000 good as get = -16396.9
std::mt19937 rvengine(seed);  // Mersenne twister random number engine
std::uniform_real_distribution<double> distr(0.0, 1.0);//call using distr(engine) to get U(0,1) variate

//std::cout<<"random U(0,1)="<<distr(engine)<<" and another "<<distr(engine)<<std::endl;

double discount = 0.9;// discounting for value function
double curQ;
unsigned int greedyA;

// set up environment which does:
// reads in data and computes necessary constants for reward (ln network score) which are independent of DAG structure so can be pre-computed
// based on observed data and prior. Default prior is alpha_w,alpha_m = 30|30. This also sets the dag, dag0 to empty dag and the position on the
// gridworld is 0,0 top left cornder and creates a unique key for this state.
envDAG env1(datafile, 0);//-6470.0);

std::cout<<"initial reward="<<env1.fitDAG()<<std::endl;//exit(1);

#ifdef globalcheck1

/** chain DAG **/
 // [a1][a2|a1][a3|a2][a4|a3][a5|a4][a6|a5][a7|a6][a8|a7][a9|a8][a10|a9]
 // my reward - fixed known DAG = -141885 with customprior1
 
arma::umat daga = { 
      {0,    0,    0,    0,    0,    0,    0,    0,    0,     0},
      {1,    0,    0,    0,    0,    0,    0,    0,    0,     0},
      {0,    1,    0,    0,    0,    0,    0,    0,    0,     0},
      {0,    0,    1,    0,    0,    0,    0,    0,    0,     0},
      {0,    0,    0,    1,    0,    0,    0,    0,    0,     0},
      {0,    0,    0,    0,    1,    0,    0,    0,    0,     0},
      {0,    0,    0,    0,    0,    1,    0,    0,    0,     0},
      {0,    0,    0,    0,    0,    0,    1,    0,    0,     0},
      {0,    0,    0,    0,    0,    0,    0,    1,    0,     0},
      {0,    0,    0,    0,    0,    0,    0,    0,    1,     0}
           };

arma::ivec posa = {0,0};// (x,y)

env1.resetDAG(daga,posa,rvengine);
std::cout<<"my reward - check DAG1 ="<<env1.fitDAG()<<std::endl;
arma::cout<<env1.dag0<<arma::endl;
exit(1);

#endif 

#ifdef globalcheck2
// [a1][a2|a1][a3][a4|a1][a5][a6|a4:a3][a7|a6][a8][a9][a10|a4:a6:a7][a11][a12][a13|a11:a12][a14][a15|a2][a16][a17][a18|a10][a19][a20|a11:a19]
// my reward - fixed known DAG =-283207 - with customprior2
arma::umat daga = { 
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,1,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0}
};

arma::ivec posa = {0,0};// (x,y)

env1.resetDAG(daga,posa,rvengine);
std::cout<<"my reward - check DAG2 ="<<env1.fitDAG()<<std::endl;
arma::cout<<env1.dag0<<arma::endl;
exit(1);

#endif 
 
#ifdef globalcheck3 
// [a1][a2][a3|a2][a4][a5|a4][a6][a7][a8][a9][a10][a11|a9][a12|a6][a13|a5:a6][a14][a15][a16|a13][a17]
//  [a18][a19|a14:a17][a20|a3:a16]
// my reward - fixed known DAG =-283197 - with customprior3
arma::umat daga = { 
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
      {0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0},
      {0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0}
};

arma::ivec posa = {0,0};// (x,y)

env1.resetDAG(daga,posa,rvengine);
std::cout<<"my reward - check DAG3 ="<<env1.fitDAG()<<std::endl;
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

unsigned int numPeriods=100;

arma::uvec stepcount(numPeriods);
bool timetostop=false;// extra hack to stop early

//arma::mat bestscores=arma::zeros<arma::mat>(numPeriods);

for(period=0;period<numPeriods;period++){
std::cout<<"PERIOD="<<period<<" ";
curDAG=dagnull;
curPos=posnull;

best_score_period=-std::numeric_limits<double>::max();//worst possible score - reset in each period

env1.resetDAG(curDAG,curPos,rvengine,false);// reset start of period to null model
//if(env1.hasCycle()){std::cout<<"ERROR - random dag has a cycle!"<<std::endl; exit(1);}
//std::cout<<"initial reward="<<env1.fitDAG()<<"->"<<env1.reward<<"->"<<env1.IsDone<<std::endl;
//std::cout<<"new pos0="<<env1.pos0(0)<<" "<<env1.pos0(1)<<std::endl;
curDagKey=env1.dagkey;// copy current dagkey
//std::cout<<"start score="<<env1.fitDAG()<<std::endl;


steps=1;
while(!env1.IsDone && steps<=400 && !timetostop)
{
//if(steps%10000==0){std::cout<<"step="<<steps<<std::endl;}

// loop through each actions and take best
best_value=-std::numeric_limits<double>::max();//worst possible reward

for(i=0;i<15;i++){
  env1.resetDAG(curDAG,curPos,rvengine);//reset back to current state including dagkey
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
                  if(curQ==best_value && distr(rvengine)>0.5){// action is same as current best action so a tie, break randomly if U(0,1)>0.5
                      best_value = curQ;
                      //if (env1.ValueMap.find(curDagKey) != env1.ValueMap.end()){std::cout<<"UPDATING EXISTING VALUE!"<<std::endl;} 
                      env1.ValueMap[curDagKey] = curQ; // update value function for just this CURRENT state
                      greedyA = i; // store best action

                  } 
                }

}

//std::cout<<"best action="<<greedyA<<std::endl<<" reward="<<env1.reward<<std::endl;
env1.resetDAG(curDAG,curPos,rvengine);// resets IsDone to false
env1.step(greedyA);// take best action i and update current state to this - this might set IsDone to true and terminate episode

if(env1.fitDAG()>bestscore){bestscore=env1.fitDAG();bestdag=env1.dag0;
  //std::stringstream().swap(temp);//clear
   temp <<period<<","<<bestscore<<std::endl; 
        
  std::cout.precision(5);
  
                            std::cout<<"best DAG score="<<std::scientific<<bestscore<<std::endl;
  std::cout.copyfmt(oldState);//restore formats
 
 /* if(bestscore> -283209){std::cout<<"reached best scores so stop!"<<std::endl;
	                timetostop=true;
 } */
  if(bestscore> -283197-1){std::cout<<"reached best score so stop!"<<std::endl;
	                timetostop=true;
			}
                          }

 if(env1.fitDAG()>best_score_period){best_score_period=env1.fitDAG();} // store the best score found during this period


//std::cout<<"current reward="<<env1.fitDAG()<<std::endl;
// now copy the current state and repeat action search
curDAG=env1.dag0; // copy - this is not efficient - fix later as no need to recreate memory
curPos=env1.pos0; // copy - this is not efficient - fix later as no need to recreate memory
curDagKey=env1.dagkey;// copy current dagkey


steps++;
} // end of episode loop/while
stepcount(period)=steps-1.0;
arma::cout<<"\t\tsteps="<<steps-1.0<<" "<<std::endl;

//periodscore_vec(period)=best_score_period;//save best score found in period into vector

//std::cout<<"\tmean="<<sum(stepcount.head(period+1))/(period+1.0)<<std::endl;
if(timetostop){std::cout<<"breaking early out of period loop"<<std::endl;
	       break;}

} // end of period loop
//arma::cout<<"stepcounts"<<arma::endl<<stepcount<<arma::endl;
str = temp.str();
std::ofstream out("output.txt");
    out << str;
    out.close();

//periodscore_vec.save("bestscoreperiods.csv", arma::csv_ascii); 

#ifdef Aa
arma::cout<<"final state="<<arma::endl<<env1.dagkey<<arma::endl;
std::cout<<"final reward="<<env1.fitDAG()<<std::endl;

std::cout<<"best score visited="<<bestscore<<std::endl;
arma::cout<<"best DAG visited="<<arma::endl<<bestdag<<arma::endl;

//print_map(env1.ValueMap);
std::cout<<"number of states stored="<<env1.ValueMap.size()<<std::endl;
#endif



/*std::ostringstream s;
for (auto const& pair: env1.ValueMap) {
        s << pair.first << "," << pair.second << std::endl;
    }
*/
//std::cout<<std::endl<<s.str();
/*
    std::ofstream out("output.txt");
    out << s.str();
    out.close();

*/


  return 0;
}
