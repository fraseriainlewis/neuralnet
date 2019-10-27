/** To compile

/Applications/MATLAB_R2019b.app/bin/mex -I/usr/local/include -L/usr/local/lib -lgsl -lgslcblas cycle.c -R2018a

*/


#include "mex.h"
void get_numincomingedges(unsigned int *incomingedges,unsigned int *graph, unsigned int numnodes);

/*#include <gsl/gsl_sf_gamma.h>*/
/* The computational routine */
void cycle(unsigned int *dag, unsigned int *graph,unsigned int *isactive, unsigned int *isactive_scratch,unsigned int *incomingedges,
        double *haveCycle, mwSize numnodes)
{
    mwSize i,j, nodesexamined,success;
 
    for (i=0; i<numnodes; i++) {isactive[i]=1;} /** all nodes initially active */
     /*mexPrintf("isactive=%u\n",isactive[0]);*/
    
    /** copy the current network definition into graph[][]*/
    for (i=0; i<numnodes; i++) {
        for (j=0; j<numnodes; j++) {
        graph[i+j*numnodes]=dag[i+j*numnodes]; /*+0.0+gsl_sf_lngamma(1.0);*/
                             }
    }
    
   /** calc number of incoming edges for each child  and put into incomingedges**/
   get_numincomingedges(incomingedges,graph, numnodes);
   /** find a node with no incoming edges - see lecture11.pdf in ../articles folder **/
nodesexamined=0;
success=1;
while(success){
        success=0;
	for(i=0;i<numnodes;i++){
        	  if(isactive[i] && !incomingedges[i]){/** if not yet examined and no incoming edges */
                	isactive[i]=0; /** set this child to inactive*/
                	
                        /** remove all OUTGOING links from node i, e.g. wherever i is a parent */
                	for(j=0;j<numnodes;j++){graph[j+i*numnodes]=0;}
                	get_numincomingedges(incomingedges,graph,numnodes);
           	        success=1; nodesexamined++;}
			}
           }
         

      if(nodesexamined==numnodes){haveCycle[0]=0.0;/*mexPrintf("no cycle");*//*return(0);*//** no cycle */                               
      } else {/*Rprintf("=>%d %d\n",nodesexamined, numnodes);*/
              haveCycle[0]=1.0;
              /*mexPrintf("yes cycle");*/} /** found a cycle */
 
  
   
   
   /* for (i=0; i<numnodes; i++) {
        for (j=0; j<numnodes; j++) {
        ycopy[i+j*numnodes]=(double)dag[i+j*numnodes];
                             }
    }
*/
    
    
}

/** *************************************************************************************************/
/** v.small function but helps clarity in hascycle() */
/** *************************************************************************************************/
void get_numincomingedges(unsigned int *incomingedges,unsigned int *graph, unsigned int numnodes)
{ /** count up how many parents each child has **/
unsigned int i,j;
unsigned int numincomedge;
for(i=0;i<numnodes;i++){/** for each child */
        numincomedge=0;
 	for(j=0;j<numnodes;j++){numincomedge+=graph[i+j*numnodes];
                                }
        incomingedges[i]=numincomedge;
        } 	

}



/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    unsigned int *inDAG,*inMatrix,*inVec1, *inVec2,*inVec3;               /* 1xN input matrix */
    size_t nnodes;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=5) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs","5 inputs required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs","One output required.");
    }

    /* make sure the first input argument is type uint32 */
    if( ! (mxIsUint32(prhs[0]) && mxIsUint32(prhs[1]) && mxIsUint32(prhs[2]) && mxIsUint32(prhs[3]) && mxIsUint32(prhs[4]))) 
         {
        mexErrMsgIdAndTxt("MyToolbox:cycle:notUint32","All inputs must be type uint32.");
    }
    
    /* check that number of rows in first input argument is 1 */
    if(mxGetM(prhs[0])!=mxGetN(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notRowVector","Input must be a square matrix.");
    }
    
    /* create a pointer to the real data in the input matrix  */
    #if MX_HAS_INTERLEAVED_COMPLEX
    inDAG = mxGetUint32s(prhs[0]);
    inMatrix = mxGetUint32s(prhs[1]);
    inVec1 = mxGetUint32s(prhs[2]);
    inVec2 = mxGetUint32s(prhs[3]);
    inVec3 = mxGetUint32s(prhs[4]);
    #else
    inMatrix = mxGetPr(prhs[0]);
    #endif

    /* get dimensions of the input matrix */
    nnodes = mxGetM(prhs[0]);

    /* create the output matrix - using 1x1 - could use scalar instead */
    plhs[0] = mxCreateDoubleMatrix((mwSize)1,(mwSize)1,mxREAL);

    /* get a pointer to the real data in the output matrix */
    #if MX_HAS_INTERLEAVED_COMPLEX
    outMatrix = mxGetDoubles(plhs[0]);
    #else
    outMatrix = mxGetPr(plhs[0]);
    #endif

    /* call the computational routine */
    cycle(inDAG,
          inMatrix,
          inVec1,/** should be all ones */
          inVec2,
          inVec3,
          outMatrix,(mwSize)nnodes);
}




