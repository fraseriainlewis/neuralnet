// c++ ffn_v1 -o a.out -std=c++11 -lboost_serialization -larmadillo -lmlpack -Wall

#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>
#include <mlpack/prereqs.hpp>

#include <sstream>

#include "cv_helper.hpp"

#define KEEP

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;

int main(int argc, char *argv[])
{
        // parse command line: ./a.out -s seed -i <maxiters> -v <varID 0-39> -a <filename> -p <filename>  
        char *avalue = NULL;// output file for actuals in each CV fold
        char *pvalue = NULL;// output file for predicted in each CV fold
        char *featurevalue = NULL;// input file of features - no header
        char *labelvalue = NULL;// input file of labels - no header
        uword maxiters = 0;// maxiters in optimizer
        uword seed =0;// rv seed
        int index;
        int c;
        
        opterr = 0;
        
        while ((c = getopt (argc, argv, "s:i:a:p:d:e:")) != -1)
                switch (c)
                {
                case 's':
                        seed = strtoul(optarg,NULL,10);
                        //std::cout<<"rv seed"<<seed<<std::endl;
                        break;  
                case 'i':
                        maxiters = strtoul(optarg,NULL,10);
                        //std::cout<<"maxiterations"<<maxiters<<std::endl;
                        break;        
                case 'a':
                        avalue = optarg;
                        break;
                case 'p':
                        pvalue = optarg;
                        break;
                case 'd':
                        featurevalue = optarg;
                        break;
                case 'e':
                        labelvalue = optarg;
                        break;
                case '?':
                        if (optopt == 's')
                                fprintf (stderr, "Option -%c requires an argument (seed).\n", optopt);
                        else if (optopt == 'a')
                                fprintf (stderr, "Option -%c requires an argument (outfile actual).\n", optopt);
                        else if (optopt == 'p')
                                fprintf (stderr, "Option -%c requires an argument (outfile predictions).\n", optopt);
                        else if (optopt == 'i')
                                fprintf (stderr, "Option -%c requires an argument (max iterations).\n", optopt);
                        else if (isprint (optopt))
                                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                        else if (optopt == 'd')
                                fprintf (stderr, "Option -%c requires an argument (input file of features).\n", optopt);
                        else if (optopt == 'e')
                                fprintf (stderr, "Option -%c requires an argument (input file of labels).\n", optopt);
                        else
                                fprintf (stderr,
                                         "Unknown option character `\\x%x'.\n",
                                         optopt);
                        return 1;
                default:
                        exit(0);
                }
        
        if(argc!=13){std::cout<<"Aborting - need 9 arguments: -s <seed> -i <maxiters> -d <infeatures> -e <inlabels> -a <outactual> -p <outpred>"<<std::endl;
                exit(0);}
        //printf ("avalue = %s, pvalue = %s\n",avalue,pvalue);
        
        
        //std::cout<<"total args="<<argc<<std::endl;
        
        for (index = optind; index < argc; index++){
                printf ("Non-option argument %s\n", argv[index]);
                exit(0);}
        
        std::cout<<"seed="<<seed <<" max iters="<<maxiters<<std::endl;            
        
        
        
/**************************************************************************************************/
/**************************************************************************************************/
// set some debugging flags 
unsigned int checkCSV=0;// print out first parts of features and labels to check these imported correctly
unsigned int checkPredict=1;// print out parts of log probabilites - the model output
// set the random number seed - e.g. shuffling or random starting points in optim
arma::arma_rng::set_seed(seed);
//arma::arma_rng::set_seed_random();


/**************************************************************************************************/
/**************************************************************************************************/
/** Load the training set - separate files for features and lables **/
/** note - data is read into matrix in column major, e.g. each new data point is a column - opposite from data file **/
arma::mat featureData, labels01;
unsigned int i,j;
//data::Load("../../03 Outputs/moistfeatures40.csv", featureData, true);// last arg is transpose - needed as boost is col major
//data::Load("../../03 Outputs/bdlabels40bin.csv", labels01, true);// binary

data::Load(featurevalue, featureData, true);// last arg is transpose - needed as boost is col major
data::Load(labelvalue, labels01, true);// binary


const arma::mat labels = labels01.row(0) + 1; // add +1 to all response values, mapping from 0-1 to 1-2 
                                              // NegativeLogLikelihood needs 1...number of classes

uword totalDataPts=featureData.n_cols;//total data set size as read in from disk


string str;          // The string
stringstream temp;  // 'temp' as in temporary

        
if(checkCSV){
        std::cout<<"#-------------------------------------------------------------------------#"<<std::endl;
        std::cout<<"#--------- Import CSV debugging on ---------------------------------------#"<<std::endl;
        std::cout<<"#-------------------------------------------------------------------------#"<<std::endl<<std::endl;
        // print out to check the data is read in correctly
        std::cout<<"\t1. Data dimensions check"<<std::endl;
        arma::cout << "\tn rows="<<featureData.n_rows <<" n cols="<< featureData.n_cols << arma::endl;
        arma::cout << "\tn rows="<< labels.n_rows <<" n cols"<< labels.n_cols << arma::endl;
        
        std::cout<<std::endl<<"\t2. First 10 Labels"<<std::endl;
        for(i=0;i<10;i++){
                arma::cout <<"\t"<< labels(0,i) << arma::endl;
        }
        
        std::cout<<std::endl<<"\t3. Features for first 2 observations"<<std::endl;
        for(j=0;j<2;j++){ std::cout<<"\t--> obs number: "<<j<<std::endl;
                for(i=0;i<featureData.n_rows;i++){
                        arma::cout <<"\t"<<featureData(i,j) << arma::endl;
                }
        }
        std::cout<<"#-------------------------------------------------------------------------#"<<std::endl;
        std::cout<<"#--------- Import CSV debugging end --------------------------------------#"<<std::endl;
        std::cout<<"#-------------------------------------------------------------------------#"<<std::endl<<std::endl;
        
} //end of if


/**************************************************************************************************/
/**************************************************************************************************/
/**************************************************************************************************/
/** MODEL DEFINITION  **/
   
// PART1 - starting weights initialize to constant
 std::cout<<"----- PART 1 ------"<<std::endl;
// initialise weights to a constant and use MSE as the loss function
FFN<NegativeLogLikelihood<>,RandomInitialization> model1(NegativeLogLikelihood<>(),RandomInitialization(-1,1));//default
// build layers
const size_t inputSize=featureData.n_rows;// 40 

// create a dropout layer - then add
Dropout<>* drop = new Dropout<>(0.5);
drop->Deterministic() = false;// must set to false, then true later when predicting

model1.Add<Linear<> >(inputSize, 20);// 40 -> 20
model1.Add<SigmoidLayer<> >();
model1.Add(drop);// this drop outs the 20 hidden layer nodes
model1.Add<Linear<> >(20, 2);// 20 -> 2
model1.Add<LogSoftMax<> >();


/**************************************************************************************************/
/************************END MODEL DEFN************************************************************/
/**************************************************************************************************/


/**************************************************************************************************/
/********************      Define Optimizer                                                 *******/
/**************************************************************************************************/
// set up optimizer 
//const size_t maxiters=20000000;  //20000000;// checked on in sample accuracy and seems enough
std::cout<<"##############--- WARNING: put hard stop on iterations = "<< maxiters<<std::endl;

ens::Adam opt(0.01,featureData.n_cols, 0.9, 0.999, 1e-8, maxiters, 1e-8,false,true); //https://ensmallen.org/docs.html#rmsprop.

// Run the model fitting on the entire available data
double lossAuto=1e+300;
arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model1.Parameters() << arma::endl;
model1.Train(featureData, labels,opt);

lossAuto=model1.Evaluate(featureData, labels);
arma::cout<<"-------final params------------------------"<<arma::endl;

// Use the Predict method to get the assignments.
arma::mat assignments;
drop->Deterministic() = true;// must set to true to use non-random network
model1.Predict(featureData, assignments);//tech remark - assignments is used later with different sizes

if(checkPredict){
        std::cout<<"#-------------------------------------------------------------------------#"<<std::endl;
        std::cout<<"#--------- log probabilities debugging on --------------------------------#"<<std::endl;
        std::cout<<"#-------------------------------------------------------------------------#"<<std::endl<<std::endl;
        
        std::cout<<"Predictions shape rows  : "<<assignments.n_rows<<std::endl;
        std::cout<<"Predictions shape cols : "<<assignments.n_cols<<std::endl;
        std::cout<<std::endl<<"\t1. selected output records, first 10 records"<<std::endl;
        
        for(i=0;i<10;i++){//assignments.n_cols;i++){
                cout<<"\t"<<i<<"\t["<<assignments(0,i)<<","<<assignments(1,i)<<"]"<<endl;
        }
        std::cout<<std::endl<<"\t2. selected output records, first 10 records which correspond to true label"<<std::endl;
        for(i=0;i<10;i++){//assignments.n_cols;i++){
                cout<<"\t"<<i<<"\t["<<assignments(labels(0,i)-1,i)<<"]"<<endl;
        }
        
        std::cout<<std::endl<<"\t3. selected output records, last 10 records"<<std::endl;
        for(i=assignments.n_cols-10;i<assignments.n_cols;i++){//assignments.n_cols;i++){
                cout<<"\t"<<i<<"\t["<<assignments(0,i)<<","<<assignments(1,i)<<"]"<<endl;
        }
        std::cout<<std::endl<<"\t4. selected output records, last 10 records which correspond to true label"<<std::endl;
        for(i=assignments.n_cols-10;i<assignments.n_cols;i++){//assignments.n_cols;i++){
                cout<<"\t"<<i<<"\t["<<assignments(labels(0,i)-1,i)<<"]"<<endl;
        }
        
} // end of prob debug check

// compute the negative log like loss manually 
double lossManual=0;
double predp2,predp1,sen,spec,acc,inSampleSe,inSampleSp,inSampleAcc;
uword P,N,TP,TN;
P=0;
N=0;
TP=0;
TN=0;
for(i=0;i<assignments.n_cols;i++){
        lossManual+= -(assignments(labels(0,i)-1,i));
        
        predp2=assignments(1,i);//log prob of being in class 2
        predp1=assignments(0,i);//log prob of being in class 1
        
        if(labels(0,i)==2){//truth is class 2 - positives
                P++;//count number of class 2
                if(predp2>=predp1){//truth is class 2 and predict class 2
                        TP++;//increment count of true positives TP
                }
        } //end of count positives
        if(labels(0,i)==1){//truth is class 1
                N++;// count number of class 1 - negatives
                if(predp1>=predp2){//truth is class 1 and predict class 1
                        TN++;//increment count of true negative FN
                }
        }
        //std::cout<<trainData.n_cols<<" "<<batchsize<<"  "<<std::setprecision(5)<<fixed<<b<<" "<<std::setprecision(2)<<fixed<<2211.0/32.0<<std::endl;          
}
sen=(double)TP/(double)P;
spec=(double)TN/(double)N;
acc=((double)TP+(double)TN)/((double)P+(double)N);//overall accuracy

std::cout<<"NLL (from Evaluate()) on full data set="<<lossAuto<<std::endl;
std::cout<<"NLL manual - and correct - on full data set="<<lossManual<<std::endl;
std::cout<<"P= "<<P<<"  TP="<<TP<<"  N= "<<N<<"  TN= "<<TN<<std::endl;
std::cout<<"in-sample sensitivity = "<<std::setprecision(5)<<fixed<<sen<<" in-sample specificity = "<<spec<<" in-sample accuracy= "<<acc<<std::endl;

inSampleSe=sen;
inSampleSp=spec;// estimates of within sampe accuracy
inSampleAcc=acc;

#ifdef KEEPa

/**************************************************************************************************/
/** we now reset the model parameters and use 10-fold cross validation to estimate the out of    **/
/** sample accuracy                                                                              **/
/*******************************CV folds **********************************************************/
/**************************************************************************************************/
uvec foldinfo;
umat indexes;
uword verbose=0;
uword nFolds=10; 
setupFolds(totalDataPts,nFolds,&foldinfo,&indexes,verbose);// pass pointers as call constructor in function

uvec trainIdx,testIdx;
uword curFold;
mat res;

double meanSe=0.0;
double meanSp=0.0;
double meanAcc=0.0;

arma::arma_rng::set_seed(seed); // re-setting seed here because this means we can repeat the folds
// in a separate cut-down program - randomness is used in traning above

for(curFold=1;curFold<=nFolds;curFold++){
        std::cout<<"Processing fold "<<curFold<<" of "<<nFolds<<std::endl;
        getFold(nFolds,curFold,indexes,foldinfo,&trainIdx,&testIdx,verbose);// note references and pointers, refs for
        
        model1.ResetParameters();// reset parameters to their initial values - should be used with clean re-start policy = true
        drop->Deterministic() = false;// implement drop-out
        arma::cout<<"-------re-start empty params------------------------"<<arma::endl;//empty params as not yet allocated
        model1.Train(featureData.cols(trainIdx), labels.cols(trainIdx),opt);
        arma::cout<<"-------re-start final params------------------------"<<arma::endl;
       
        // Save the actual data labels in the current fold
        res = (labels.cols(testIdx)).t();
        std::stringstream().swap(temp);//clear
        //temp << "../../03 Outputs/CV10seed100004/1layer20DCVactual"<<curFold<<".csv"; 
        temp << avalue<<curFold<<".csv"; 
        str = temp.str();
        std::cout<<str<<std::endl;
        res.save(str, csv_ascii); 
        
        // Use the Predict method to get the assignments.
        // need to turn on deterministic - turns off dropout
        drop->Deterministic() = true;
        model1.Predict(featureData.cols(testIdx), assignments);
        // Save the predicted labels in the current fold
        res = assignments.t();
        std::stringstream().swap(temp);//clear
        //temp << "../../03 Outputs/CV10seed100004/1layer20DCVpred"<<curFold<<".csv"; 
        temp << pvalue<<curFold<<".csv"; 
        str = temp.str();
        std::cout<<str<<std::endl;
        res.save(str, csv_ascii);   
        
        
        std::cout<<"SIZE="<<assignments.n_cols<<std::endl;
        
        P=0;
        N=0;
        TP=0;
        TN=0;
        sen=0.0;spec=0.0;acc=0.0;
        // compute the negative log like loss manually 
        lossManual=0.0;
        for(i=0;i<assignments.n_cols;i++){
                predp2=assignments(1,i);//log prob of being in class 2
                predp1=assignments(0,i);//log prob of being in class 1
                
                if(labels(0,testIdx(i))==2){//truth is class 2 - positives
                        P++;//count number of class 2
                        if(predp2>=predp1){//truth is class 2 and predict class 2
                                TP++;//increment count of true positives TP
                        }
                } //end of count positives
                
                if(labels(0,testIdx(i))==1){//truth is class 1
                        N++;// count number of class 1 - negatives
                        if(predp1>=predp2){//truth is class 1 and predict class 1
                                TN++;//increment count of true negative FN
                        }
                }
                
        }
        
        sen=(double)TP/(double)P;
        spec=(double)TN/(double)N;
        acc=((double)TP+(double)TN)/((double)P+(double)N);//overall accuracy
        
        meanSe+=sen;
        meanSp+=spec;
        meanAcc+=acc;
        
        std::cout<<"P= "<<P<<"  TP="<<TP<<"  N= "<<N<<"  TN= "<<TN<<std::endl;
        std::cout<<"sensitivity = "<<std::setprecision(5)<<fixed<<sen<<" specificity = "<<spec<<" accuracy= "<<acc<<std::endl;
        
} //end of fold loop

// output overall mean sen and spec
std::cout<<std::endl<<std::endl<<"in-sample Se= "<<std::setprecision(5)<<fixed<<inSampleSe<<" in-sample Sp = "<<inSampleSp<<
        " in-sample Acc = "<<inSampleAcc<<std::endl;
std::cout<<"out-sample 10-fold mean Se = "<<std::setprecision(5)<<fixed<<meanSe/(double)nFolds<<" out-sample 10-fold mean Sp = "<<
        meanSp/(double)nFolds<< " out-sample 10-fold mean accuracy = "<<meanAcc/(double)nFolds<<std::endl<<std::endl;


#endif

}
