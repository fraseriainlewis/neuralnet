/* ***********************************************************************
envDAG.cpp define a class for fitting and searching across DAGs
designed to be the environment driven by an agent in RL learning


***********************************************************************/
#include "envDAG.hpp"
#include <iostream>
#include <iomanip>
#include <armadillo>
#include <gsl/gsl_math.h>

#define DEBUG

envDAG::envDAG(const std::string datafile, const double _alpha_w, const double _alpha_m): alpha_w(_alpha_w), alpha_m(_alpha_m) { 
    /** constructor
    @datafile = string of the filename which has the data against which each DAG is fitted. All columns are used, and in order provided
    @alpha_w > n-1 hyperparameter for the Wishart part of Normal-Wishart, n = dimension of DAG
    @alpha_m > 0  hyperparameter for the Normal part of Normal-Wishart
    */

    /********************/
    /** 1. read in the raw data from disk - CSV without header line **/
	bool ok = rwdata.load(datafile);
    if(ok == false){
    	std::cout << "problem with loading raw data from file="<<datafile<<std::endl;
    	exit(-1);
    }

    /********************/
    /** 2. compute useful constants used across all DAG fitting so compute once first */
   
    n = rwdata.n_cols; //number of variables - dimension of DAG
    N = rwdata.n_rows; //number of obs

    nu = arma::ones<arma::umat>(1,n);  //prior precision for each node = 1
    mu0 = arma::zeros<arma::mat>(1,n); //prior intercept for each node - all zero
    b = arma::zeros<arma::mat>(n,n);   //prior regression coefs - all zero
   
    #ifdef heckprior
   /* to use the informative prior coefficients as in the example in 3.5 in Heckerman 1994 define heckprior, also need to 
      provide the data given in heckerman example see Table 1 in heckermand 94 */
    mu0(0,0) = 0.1;
    mu0(0,1) = -0.3;
    mu0(0,2) = 0.2;
  
    b(0,2)=1;
    b(1,2)=1;              
    #endif

    // Compute T = prior precision matrix in Wishart prior.
    // See function defn for explanation 
    envDAG::setT();// sets matrix T
    
    // Compute R = updated T with data  
    envDAG::setR();// sets matrix R
    
    /********************/
    /** 3. setup initial DAG and DAG related graph helpers */
    
    /** create and set empy dag **/
    dag0 = arma::zeros<arma::umat>(n,n);// dag0 is always the current DAG 

    /** to specify a custom dag for testing could do it here, e.g. dag0(2,3)=1; etc **/ 
    /** initialize position on board - here 0,0 top left corner */
    arma::cout<<"pos="<<arma::endl<<envDAG::pos0<<arma::endl;

    // set up actions 
    #ifdef AA
    actionLookup={            [0 1],[0 0],[0 -1],... % no spatial move 
                              [1 1],[1 0],[1 -1],...             % left
                              [2 1],[2 0],[2 -1],...             % right
                              [3 1],[3 0],[3 -1],...             % up
                              [4 1],[4 0],[4 -1]};
    #endif
    
    /** actions are a matrix, each row is an action and the cols are the components of the action */
    actions={ {0,1},{0,0},{0,-1}, // no spatial move, add,do nothing,remove arc
              {1,1},{1,0},{1,-1}, // left
              {2,1},{2,0},{2,-1}, // right
              {3,1},{3,0},{3,-1}, // up
              {4,1},{4,0},{4,-1}  // down
             };

    //arma::cout<<"actions="<<actions<<arma::endl;

    /** these below are setting up storage to help with cycle checking - fixed size to allocate once **/
    isactive=arma::zeros<arma::uvec>(n);
    isactive_scratch=arma::zeros<arma::uvec>(n);
    incomingedges=arma::zeros<arma::uvec>(n);
    graph=arma::zeros<arma::umat>(n,n);


} //end of constructor


/*****************************************************/
/** METHODS: take a step **/
/*****************************************************/
void envDAG::step(const unsigned int actidx){
 
//arma::cout<<"action passed is="<<actions.row(actidx)<<arma::endl;

//arma::cout<<"current position (x,y) is=("<<pos0(0)<<","<<pos0(1)<<")"<<arma::endl;
// now update position on board - dag
int pos_act=actions(actidx,0);// position_action - where do we move to
//std::cout<<"pos action passed="<<pos_act<<std::endl;

int r=pos0(0);// (row,col) row coord 
int c=pos0(1);// (row,col) col coord

 dag_cp=dag0;// a copy - in case we need to revert back because move introduces a cycle

 //note origin is top left corner 0,0, so 1,2 is one row along and two rows down
 switch(pos_act) {
    case 0: { // no spatial move do nothing
            break;
            }
    case 1: { // left spatial move
            std::cout<<"left"<<std::endl; 
    	    if(c==0){// at left edge so can't move any further left so do not update
    	      } else {c--; }//decrement x coord 
            break;
            }    
    case 2: { // right spatial move
            std::cout<<"right"<<std::endl;  
    	    if(c==(n-1)){// at right edge so can't move any further right so do not update
    	      } else {c++; }//increment x coord 
            break;
            }  
    case 3: { // up spatial move 
    	    std::cout<<"up"<<std::endl; 
    	    if(r==0){// at top edge so can't move any further up so do not update
    	      } else {r--; }//increment y coord 
            break;
            } 
    case 4: { // down spatial move 
    	    std::cout<<"down"<<std::endl; 
    	    if(r==(n-1)){// at bottom edge so can't move any further down so do not update
    	      } else {r++; }//increment y coord 
            break;
            } 

    default: 
             // should never get here!
             std::cout << "switch pos_act ERROR!\n";
             exit(1);
   } //end of switch

 if(r>=n || r<0 || c>=n || c<0){//boundary check 
                               std::cout<<"Boundary error - we have moved off the board!!"<<std::endl;
                               exit(-1);}

 //do updates of position
 pos0(0)=r;
 pos0(1)=c;

 //
 
 
int arc_act=actions(actidx,1);// act action do we add/nothing/remove arc 

	switch(arc_act){
		case 0:{ // no arc change so do nothing
            break;
            }
        case 1:{ // add an arc change at the current position from the first part of action above
            dag0(r,c)=1; // at row r and col c 
            break;
            }
        case -1:{ // remove an arc change at the current position from the first part of action above
            dag0(r,c)=0; // at row r and col c 
            break;
        default: 
             // should never get here!
             std::cout << "switch arc_act ERROR!\n";
             exit(1);       }

	} 

arma::cout<<"new position (r,c) is=("<<pos0(0)<<","<<pos0(1)<<")"<<arma::endl;
arma::cout<<"new dag="<<arma::endl<<dag0<<arma::endl;

}


/*****************************************************/
/** METHODS: preparation - computation of constants **/
/*****************************************************/

void envDAG::setT(void){
// Compute T = prior precision matrix in Wishart prior.

 arma::uword wsize=1;
 
 arma::mat w1  = arma::zeros<arma::mat>(wsize,wsize);
 arma::mat w2,e1,sigmainv;
 unsigned int i=0;

 w1(0,0) = 1/nu(0,0);// initial matrix - see w1+kron below
 // w1 is current matrix
 // w2 is new matrix, then copy over w1 etc itatatively 

 // see equations 5 and 6 in Geiger and Heckerman 1994. Fiddly but checked output against example 3.5 in Heckerman 94
 for(i=1;i<n;i++){
  	wsize=i;
  	w2  = arma::zeros<arma::mat>(wsize+1,wsize+1); // initialize
  	e1=b.submat(0,i,(i-1),i);// row 0:(i-1), col i 
    //e1=e2;
  	w2.submat(0,0,(wsize-1),(wsize-1)) = w1+kron(e1,e1.t())/nu(0,i); // set submatrix top corner
  	w2.submat(wsize,0,wsize,(wsize-1))= -e1.t()/nu(0,i); //last row
  	w2.submat(0,wsize,(wsize-1),wsize)= -e1/nu(0,i);  // last col
  	w2(wsize,wsize) = 1/nu(0,i); // bottom right cell

  	w1=w2; // update copy new matrix to current matrix
 }
 
 sigmainv=w2;// prior precision matrix for Wishart

 //We need precision matrix T which defines the prior Wishart distribution. 
 //Basic method: Equation 20 in 2002 defines the covariance of X as a function of (T^prime)^-1 we know the cov of X, it's just inv(sigmainv) from Equation 5. 
 //so we have the left hand side of Equation 20. Now equation 19 gives us an expression for T^prime = RHS, and so (T^prime)^-1 is just the inverse of the RHS of equation 19
 //which reduces to inv(sigmainv) = (alpha_w-n+1)/(alpha_w-n-1)*(alpha_m+1)/(alpha_m*(alpha_w-n+1)) * T, and cancelling the terms and 
 //re-arranging gives T = inv(sigmainv)/sigmaFactor as below. Lots of faff but easy enough.   
 double sigmaFactor = (alpha_m+1)/(alpha_m*(alpha_w-n-1));
 T =inv(sigmainv)/sigmaFactor; // this matches the T0 matrix values given in 1994 Heckerman - so works ok

}


void envDAG::setR(void){  
// Compute R = precision matrix term in network/node score expression - update T with data
// NOTE using equation A.15 in Kuipers et al 2014 in the SI which is same as from Heckerman 1994 (eqn 9) and Heckerman 2003 (eqn 17) - note as per email with Kuipers in Sept
// the equation 4 in Kuipers main manuscript incorrecly has alpha_w rather than alpha_m - should be as below, alpha_m

 arma::mat xbarL=mean(rwdata);//col - variable - means
 // this part computes the equivalent of an outer multiplication - (mu0-xbarL).*(mu0-xbarL)' in matlab col vec * row vec element at a time
 arma::mat tmp1 = (mu0-xbarL); // row vector
 arma::mat tmp2 = tmp1.t();//col vector
 arma::mat tmp  = arma::zeros<arma::mat>(n,n);
 unsigned int i=0;
 for(i=0;i<n;i++){tmp.col(i)=(tmp1*tmp2(i,0)).t();}

 R = T + cov(rwdata)*(N-1) + (alpha_m*N)/(alpha_m+N) * tmp;

 // xbarL=mean(thedata); //matlab equivalent code
 // R = T + cov(thedata)*(N-1) + (alpha_m*N)/(alpha_m+N) * (mu0-xbarL).*(mu0-xbarL)';
 // this matches matrix values given in 1994 Heckerman - but this because alpha_w=alpha_m in that example. 

}

/*****************************************************/
/** METHODS: checking cycles/fitting DAG/compute network scores **/
/*****************************************************/

/* The computational routine for checking for cycles */
bool envDAG::hasCycle(const arma::umat dag)
{   

    unsigned int i,j, nodesexamined,success;
    isactive.ones();//reset to all 1
    //arma::cout<<"isactive FIRST"<<arma::endl<<isactive<<arma::endl;
    graph=dag;// copy the current network definition into graph[][]
    
   /** calc number of incoming edges for each child  and put into incomingedges**/
   envDAG::get_numincomingedges();
   /** find a node with no incoming edges - see lecture11.pdf in ../articles folder **/
nodesexamined=0;
success=1;
while(success){
        success=0;
	for(i=0;i<n;i++){
    //arma::cout<<"node i"<<i<<arma::endl;
        	  if(isactive(i) && !incomingedges(i)){/** if not yet examined and no incoming edges */
                	isactive(i)=0; /** set this child to inactive*/
                	//arma::cout<<"isactive="<<i<<arma::endl<<isactive<<arma::endl;
                        /** remove all OUTGOING links from node i, e.g. wherever i is a parent */
                	for(j=0;j<n;j++){graph(j,i)=0;}
                    //std::cout<<std::endl<<"update graph"<<std::endl;
                	envDAG::get_numincomingedges();
           	        success=1; nodesexamined++;}
           	        //std::cout<<"nodesexamined="<<nodesexamined<<std::endl;  
			}
           }
        

      if(nodesexamined==n){return(false);/*mexPrintf("no cycle");*//*return(0);*//** no cycle */                               
      } else {/*Rprintf("=>%d %d\n",nodesexamined, numnodes);*/
              return(true);
              /*mexPrintf("yes cycle");*/} /** found a cycle */  
    
}

/*****************************************************/
/*****************************************************/

void envDAG::resetDAG(const arma::umat dag, const arma::ivec pos)
{ // set network definition and board locationb

	dag0=dag;// no checks yet - needed?
    pos0=pos;// no checks
    arma::cout<<"Reset position to="<<arma::endl<<pos0<<arma::endl<<"Reset DAG to="<<arma::endl<<dag0<<arma::endl;

}


#ifdef old 
	/*****************************************************/
	void envDAG::fitDAG(const arma::umat dag){
	// pass a DAG and compute network score - NOTE: this clobbers private member dag0 - so be aware

	dag0=dag;//copy
	if(!envDAG::hasCycle()){
		std::cout<<"no cycle :-)"<<std::endl;
		envDAG::fitDAG();
		} else {arma::cout<<"has cycle!!!"<<arma::endl<<dag0<<arma::endl;
        	    //exit(1);
            	}
}
#endif
/*****************************************************/
void envDAG::fitDAG(void){
// compute network score using current member dag0 

arma::uword nrow=dag0.n_rows; // n is dimension
double totLogScore=0.0;    
unsigned int i,j;
arma::umat nodei;//move this to private member? to allowpreallocation
arma::uvec par_idx,YY;
arma::uword npars=0;
arma::uword l=0;
double A,B;

for(i=0;i<nrow;i++){
    // process node i
    nodei=dag0.row(i);
    //arma::cout<<"nodei="<<nodei<<arma::endl;
    par_idx=arma::find(nodei);//% indexes of parents of node i
    npars=par_idx.n_elem;
    //std::cout<<"num pars="<<npars<<std::endl;
    if(npars==0){ 
      //std::cout<<"no parents"<<std::endl;
      // we are done as p(d) = singleX/1.0
      l=npars+1;//l=dimension of d - note here npars=0 always, just so same line can be used below
      YY.set_size(l);
      YY(0)=i;
      //totLogScore=totLogScore+pDln(N,n,l,alpha_m,alpha_w,T,R,YY);
      //std::cout<<"nodescore for node="<<i<<pDln(l,YY)<<"l="<<l<<"YY="<<YY<<std::endl;

      totLogScore = totLogScore + pDln(l,YY);

     } else { // we have parents
     	//std::cout<<"parents"<<std::endl;

             l=npars+1;//l=dimension of d - note here npars=0 always, just so same line can be used below
             // we want to do matlab equivalent YY=[par_idx i], where par_idx is a vector - 
             YY.set_size(l);
             for(j=0;j<l-1;j++){YY(j)=par_idx(j);}
             YY(l-1)=i;
             A=pDln(l,YY);   
         // repeat as above but with current node in YY - matlabe equivalent YY=[par_idx];
             l--;//decrement l as one less node in calc
             YY.set_size(l);
             for(j=0;j<l;j++){YY(j)=par_idx(j);}
             B=pDln(l,YY);
             totLogScore=totLogScore+A-B;
         
         }


} // end of for over nodes 



lnscore = totLogScore;

arma::cout<<"this is fitdag"<<arma::endl<<" dag0="<<arma::endl<<dag0<<arma::endl<<"lnscore=" << std::setprecision(6) << std::scientific<<lnscore<<arma::endl;

//arma::cout<<arma::endl<<"lnscore=" << lnscore<<arma::endl;

}

double envDAG::pDln(const unsigned int l, const arma::uvec YY)
{
 double term1,term2,term3,topGamma,botGamma,topdet,botdet;

 term1 = (l/2.0)*log(alpha_m/(N+alpha_m));
 //std::cout<<"pdln="<<(l/2.0)*log(alpha_m/(N+alpha_m))<<"alpha_m="<<alpha_m<<std::endl;
 
 topGamma=gammalnmult(l,(N+alpha_w-n+l)/2.0);
 botGamma=gammalnmult(l,(alpha_w-n+l)/2.0);

 term2 = topGamma-botGamma-(l*N/2)*log(M_PI);

double val;
double sign;
log_det(val, sign, T(YY,YY)); 

//topdet = ((alpha_w-n+l)/2.0)*log(det(T(YY,YY)));
if(sign<0){std::cout<<"WARNING NEGATIVE TOPDET!!"<<std::endl;}
topdet = ((alpha_w-n+l)/2.0)*val;

log_det(val, sign, R(YY,YY)); 
if(sign<0){std::cout<<"WARNING NEGATIVE BOTDET!!"<<std::endl;}
botdet = ((N+alpha_w-n+l)/2.0)*val;
//botdet = ((N+alpha_w-n+l)/2.0)*log(det(R(YY,YY)));
term3 = topdet-botdet; 

//std::cout<<std::endl<<"term1="<<term1<<" term2="<<term2<<" term3="<<term3<<" topdet="<<det(T(YY,YY))<<" botdet="<<botdet<<std::endl;

	return(term1+term2+term3);// the complete DAG score term
}

double envDAG::gammalnmult(const double l, const double xhalf){
 
 double x,myfactor,prod1;
 unsigned int j;
 x=2.0*xhalf; // conver to function of x not x/2
 myfactor=(l*(l-1)/4.0)*log(M_PI);
 prod1=0.0; // initial value - identity for sum
 for(j=1;j<=l;j++){
   prod1=prod1+lgamma( (x+1-j)/2.0 );
 }
 ////res = myfactor + sum(gammaln( (x+1-[1:l])/2 ));


  return(myfactor+prod1);

}


/** *************************************************************************************************/
/** v.small function but helps clarity in hascycle() */
/** *************************************************************************************************/
void envDAG::get_numincomingedges()
{  //operates on member graph

/** count up how many parents each child has **/
unsigned int i;

arma::umat nodei;//move this to private member? to allowpreallocation
arma::uvec par_idx;

//arma::cout<<"graph="<<i<<arma::endl<<graph<<arma::endl;

for(i=0;i<n;i++){/** for each child node */
//std::cout<<"node="<<i<<std::endl;
    nodei=graph.row(i);// get all possible parents vector
  //  arma::cout<<"nodei="<<arma::endl<<nodei<<arma::endl;
    par_idx=find(nodei);// find how many are actually parents - non zero entries
    incomingedges(i)=par_idx.n_elem;// number of parents
    //arma::cout<<"here=i "<<i<<arma::endl<<incomingedges<<" elem="<<par_idx.n_elem<<arma::endl;
	}

//arma::cout<<"incoming="<<arma::endl<<incomingedges<<arma::endl;

}




