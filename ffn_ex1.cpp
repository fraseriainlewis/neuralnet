#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>
#include <mlpack/prereqs.hpp>


using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;

int main()
{

// set the random number seed - e.g. shuffling or random starting points in optim
arma::arma_rng::set_seed(100);
//arma::arma_rng::set_seed_random();

/**************************************************************************************************/
/** Load the training set - separate files for features and lables (regression) **/
/** note - data is read into matrix in column major, e.g. each new data point is a column - opposite from data file **/
arma::mat trainData, trainLabels;
uword i,j;
data::Load("features.csv", trainData, true);
data::Load("labelsNL1.csv", trainLabels, true);// regression response

// print out to check the data is read in correctly
arma::cout << "n rows="<<trainData.n_rows <<" n cols="<< trainData.n_cols << arma::endl;
arma::cout << "n rows="<< trainLabels.n_rows <<" n cols"<< trainLabels.n_cols << arma::endl;

std::cout<<"First 10 Labels:"<<std::endl;
for(i=0;i<10;i++){
 arma::cout << trainLabels(0,i) << arma::endl;
}

j=0;
std::cout<<"Features for observation: "<<j<<std::endl;
for(i=0;i<trainData.n_rows;i++){
 arma::cout << trainData(i,j) << arma::endl;
}
j=1;
std::cout<<"Features for observation:"<<j<<std::endl;
for(i=0;i<trainData.n_rows;i++){
 arma::cout << trainData(i,1) << arma::endl;
}
/**************************************************************************************************/
/** MODELS - simple linear regression examples using different optim and start criteria **/

// PART1 - starting weights initialize to constant
std::cout<<"----- PART 1 ------"<<std::endl;
// initialise weights to a constant and use MSE as the loss function
FFN<MeanSquaredError<>,RandomInitialization> model1(MeanSquaredError<>(),RandomInitialization(-1,1));
// build layers
const size_t inputSize=trainData.n_rows;// 9 
const size_t outputSize=1;
const size_t hiddenLayerSize=2;

model1.Add<Linear<> >(inputSize, hiddenLayerSize);
model1.Add<SigmoidLayer<> >();
model1.Add<Linear<> >(hiddenLayerSize, outputSize);
//model1.Add<IdentityLayer<> >();

//Adam optimizer(0.001, 32, 0.9, 0.999, 1e-8, 100000, 1e-5, false,true);

// set up optimizer 
ens::Adam opt(0.001, 32, 0.9, 0.999, 1e-8, 0, 1e-5,false,true); //https://ensmallen.org/docs.html#rmsprop.
                 // 2nd arg is batch size,
                 // 5th arg is max iterations (0=no limit) 
                 // 6th is tolerance 
                 // 7th shuffle = false
                 // 8th clean re-start policy = true

arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model1.Parameters() << arma::endl;
model1.Train(trainData, trainLabels,opt);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;


// Use the Predict method to get the assignments.
arma::mat assignments;
model1.Predict(trainData, assignments);

//cout<<"Predictions    : "<<assignments<<endl;
//cout<<"Correct Labels : "<<trainLabels<<endl;
// print out to check the data is read in correctly
double loss=0;
for(i=0;i<assignments.n_cols;i++){
        loss+= (assignments(0,i)-trainLabels(0,i))*(assignments(0,i)-trainLabels(0,i));
}
//loss=loss/ (double)assignments.n_cols;
arma::cout<<"MSE="<<loss<<arma::endl;
arma::cout << "n rows="<<assignments.n_rows <<" n cols="<< assignments.n_cols << arma::endl;


//data::Save("model.xml", "model", model1, false);



}
