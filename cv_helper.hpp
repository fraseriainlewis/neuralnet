
using namespace arma;
using namespace std;


/***********************************************************************************************************/
/***********************************************************************************************************/
/** Function definitions                                                                                   */
/***********************************************************************************************************/
/***********************************************************************************************************/
// purpose: perform setup needed before K-fold training/evaluation 
void setupFolds(uword totalDataPts, uword nFolds, uvec* helpers,umat* out,uword verbose){
  /** pass: totalDataPts = total number of rows in data (test+training)
   *        nFolds = number of folds, i.e. 10
   *        helper = storage and after call will contain number of items in a fold and in a last (possibly) ragged fold
   *        out = the main results, after call will hold a matrix of indexes
   *        verbose = 0 no debug output, 1 lots of std::cout info
   */
  
  if(verbose){std::cout<<"----------- Verbose mode on for kfold indexes computation -------------"<<std::endl<<std::endl;
    std::cout<<"Total number of data points="<<totalDataPts<<" Number of folds="<<nFolds<<std::endl;
  }
  /** 1. get random shuffle of indexes - each index is a data point in full data set */
  uvec dataIdx = regspace<uvec>(0, 1,totalDataPts-1);  // 0, 1,...,n-1
  uvec dataIdxRv = shuffle(dataIdx);// 2,0,1,30,45,...
  
  /** 2. compute foldSize and lastFoldSize */
  uword foldSize,lastFoldSize;// Yields an integer
  
  if(totalDataPts%nFolds==0){// total data size can be equally split across folds
    foldSize=totalDataPts/nFolds;//integer division
    lastFoldSize=foldSize;// same a foldsize i.e. same as for all other folds
    std::cout<<"data is exact multiple of number of folds "<<foldSize<<" lastFoldSize "<<lastFoldSize<<std::endl;
  } else { //data does not split equally across folds ragged situation where size of last fold will be larger than the other folds
    foldSize = totalDataPts/nFolds; // integer division so any fraction dropped
    lastFoldSize =totalDataPts-(nFolds-1)*foldSize;// this is how many points are left so all are used
    if(verbose){std::cout<<"fold size="<<foldSize<<" last fold size="<<lastFoldSize<<std::endl;}
    
  }
  /** 3. create a matrix with lastFoldSize rows and nFolds columns where if lastFoldSize>foldSize
   * then the last rows in all but the last col will be "empty" due to ragged last fold. Note we
   * use "empty" to mean each cell has value uword totalDataPts - this is not permitted to be used 
   * as an index later as will give an out of bounds error
   */
  umat CVidx(lastFoldSize,nFolds); CVidx.fill(totalDataPts);//create matrix and fill it
  // we iterate over the shuffled vector of data point indexes, from end to end and unroll these
  // row by row into a rectangular matrix (CVidx) with nFolds columns. Ragged part makes this 
  // more complicated. Inside the iteration loop seems more complex than it should be, but this has been
  // checked and it works as it should, if logically ugly. Also was prototyped with 1-based indexing
  // which is why there are mixed of +1 then later -1 adjustments. 
  
  uvec::iterator it     = dataIdxRv.begin();
  uvec::iterator it_end = dataIdxRv.end();
  
  uword mycol=1;// initialize - essential in iterator loop
  uword lastFoldRag=0;// initialize - essential in iterator loop
  uword myrow=0;// initialize - essential for next loop
  uword j=1;// initialize - essential for next loop
  
  if(verbose){arma::cout<<std::endl<<"----check rolling of shuffled index vector into matrix--"<<arma::endl<<std::endl;}
  
  for(; it != it_end; ++it)  // for each index (a single data point)
  {
    if(mycol<=nFolds && lastFoldRag==0){// is active until we get to the ragged part of the last fold
      
      myrow=((j-1)%foldSize)+1; // compute row index in matrix CVidx where to copy shuffled index to 
      // modulo to map original indexes into ranges within foldsize, +1 is 1-based indexing
      if(verbose){std::cout<<"row= "<<myrow-1<<" j= "<<j-1<<"\tcol= "<<mycol-1<<"\tlastFoldRagged= "<<lastFoldRag<<std::endl;}
      CVidx(myrow-1,mycol-1)=dataIdxRv(j-1); //do actual copy - note -1, -1 as moved to 0-based indexing
      
    } else {// if in last fold and at the first ragged entry
      myrow=((j-1)%foldSize)+1+foldSize; // as above but with +foldsize since we need to exceed foldsize modulo
      // due to ragged, e.g. if fold is size 40 the last fold might be size 43
      mycol=nFolds;// this rolls back mycol from nFolds+1 back to nFolds, otherwise beyond end of array
      // rolls over end of array because increments on foldSize modulo below. 
      lastFoldRag=1; // flag to say we are in the ragged part, and so sets first if() to false
      if(verbose){std::cout<<"row= "<<myrow-1<<" j= "<<j-1<<"\tcol= "<<mycol-1<<"\tlastFoldRagged= "<<lastFoldRag<<std::endl;}
      CVidx(myrow-1,mycol-1)=dataIdxRv(j-1); //do actual copy - note -1, -1 as moved to 0-based indexing
    }
    if(j%foldSize==0){mycol=mycol+1;} //increment col - this is clobbered above but just for iterations in last col
    
    j++;//outer index
  }    
  
  if(verbose){arma::cout<<std::endl<<"---------CVidx matrix of indexes-------------------------------------------------------"<<arma::endl;
    arma::cout<<"\n\n\n"<<CVidx<<arma::endl;
    arma::cout<<"---------end of creating matrix of nFolds cols x ragged rows of indexes-----------------"<<arma::endl;}
  
  uvec vars(2); vars(0)=foldSize;vars(1)=lastFoldSize;
  
  (*helpers)=vars;// does this result in a memory leak with repeated function calls? Valgrind says not. unsure. 
  (*out)=CVidx; // does this result in a memory leak with repeated function calls? Valgrind says not. unsure. 
  // assume destructor may not be called because pointer in outside scope, but then it later points to something else
  // a to-do. works fine as is in terms of correct results
  
  
}

/***********************************************************************************************************/
/***********************************************************************************************************/
// return two vectors of indexes, one to get training case and one for test cases 
void getFold(uword nFolds,uword foldNum,umat CVidx, uvec helper, uvec* kIdxtrain, uvec* kIdxtest, uword verbose){
  /** pass: nFolds    = number of folds, e.g. 10
   foldNum   = which test fold is needed, note the last test fold could be ragged
   CVidx     = reference to matrix of indexes created in setupFolds()    
   helper    = reference to vector of useful values computed in setupFolds(), fold sizes 
   kIdxtrain = pointer and after call will point to all indexes for the training data
   kIdxtest  = pointer and after call will point to all indexes for the test data
   verbose   = flag to turn on or off debugging output to std::cout
   */
  
  /** Overview. CVidx has cols which are folds and in each row a random index of the data point in that fold 
   TEST set comprises of 1 col, TRAINING set is all the other cols 
   general approach is grab the col in CVidx for the test test, trimming off any extra values from the last rows, this is kIdxtest
   *  then drop this col from CVidx, and then grab all the remaining cols except last also trimming off any extra values, =CVidxtmp2, then
   *  grab the last col in the already col-dropped CVidx, this is the potentially ragged col, and now grab all the values in this
   *  =CVidxtmp3, final step is to flatten CVidxtmp2, then join to CVidxtmp3, this gives kIdxtrain.
   *  The else() is because it is slightly easier/different when the test fold is the ragged one, as less steps. 
   */
  
  uword foldSize=helper(0); 
  uword lastFoldSize=helper(1);
  uword curTestFold=foldNum;
  //uword curTestFold=1;//fold number, 1-indexed
  //uvec kIdxtest,kIdxtrain;// this will hold the indexes for the test set, and training set
  umat CVidxtmp,CVidxtmp2,CVidxtmp3;// temp holders
  uvec vec_CVidxtmp2,vec_CVidxtmp3;//temp holders
  
  if(curTestFold<nFolds){ //test fold is not the last fold so we know not ragged fold
    //get test indexes for current fold
    (*kIdxtest) = CVidx(span(0,foldSize-1),curTestFold-1);// col of indexes for data points in kth fold data set
    // get train indexes for current fold - i.e. all the remaining folds
    CVidxtmp = CVidx;//make a copy of CVidx as going to drop a col in this
    CVidxtmp.shed_col(curTestFold-1);// drop col for current fold
    CVidxtmp2 = CVidxtmp(span(0,foldSize-1),span(0,CVidxtmp.n_cols-2));// grab indexes in all cols EXCEPT the last
    CVidxtmp3 = CVidxtmp(span(0,lastFoldSize-1),CVidxtmp.n_cols-1);// grab indexes in the last - possibly ragged fold
    vec_CVidxtmp2=vectorise(CVidxtmp2);//flatten to single col
    (*kIdxtrain)=join_cols( vec_CVidxtmp2,  CVidxtmp3 ); //join into vector - now have a col of all the indexes needed for training
    
  } else {//test set is the last fold and so possibly ragged
    //get test indexes for current fold
    (*kIdxtest) = CVidx(span(0,lastFoldSize-1),curTestFold-1);// col of indexes for data points in kth fold data set
    // get train indexes for current fold - i.e. all the remaining folds
    CVidxtmp = CVidx;//make a copy of CVidx as going to drop a col in this
    CVidxtmp.shed_col(curTestFold-1);// drop col for current fold - the last col in CVidxtmp
    CVidxtmp2 = CVidxtmp(span(0,foldSize-1),span(0,CVidxtmp.n_cols-1));// grab valid row indexes in all cols
    (*kIdxtrain)=vectorise(CVidxtmp2);//flatten
  }
  
  
  if(verbose){arma::cout<<std::endl<<"---------KIdxtest vector of test indexes: fold="<<curTestFold<<"---------------------------"<<arma::endl;
    arma::cout<<"test length="<<(*kIdxtest).n_elem<<" ncols= "<<(*kIdxtest).n_cols<<" nrows="<<(*kIdxtest).n_rows<<arma::endl;
    arma::cout<<(*kIdxtest)<<arma::endl<<arma::endl;
    arma::cout<<std::endl<<"---------KIdxtrain vector of training indexes: fold="<<curTestFold<<"------------------------------"<<arma::endl;
    arma::cout<<"train length="<<(*kIdxtrain).n_elem<<" ncols= "<<(*kIdxtrain).n_cols<<" nrows="<<(*kIdxtrain).n_rows<<arma::endl;
    arma::cout<<(*kIdxtrain)<<arma::endl<<arma::endl;
  }
  
  if(verbose){arma::cout<<arma::endl<<"---------end of creating kfolds indexes-----------------"<<arma::endl;}
  
}