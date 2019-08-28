#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/loss_functions/mean_squared_error.hpp>
#include <mlpack/methods/ann/init_rules/const_init.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <ensmallen.hpp>
#include <mlpack/prereqs.hpp>

#define DEBU

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;

int main()
{

// set the random number seed - e.g. shuffling or random starting points in optim
arma::arma_rng::set_seed(1000);//1000
//arma::arma_rng::set_seed_random();

/**************************************************************************************************/
/** Load the training set - separate files for features and lables (regression) **/
/** note - data is read into matrix in column major, e.g. each new data point is a column - opposite from data file **/
arma::mat trainData, trainLabels;
uword i,j;
data::Load("data/AEdata.csv", trainData, true);//  
data::Load("data/AEdata.csv", trainLabels, true);// use same data as train and test is the same in AE

// print out to check the data is read in correctly
arma::cout << "n rows="<<trainData.n_rows <<" n cols="<< trainData.n_cols << arma::endl;
//arma::cout << "n rows="<< trainLabels.n_rows <<" n cols"<< trainLabels.n_cols << arma::endl;

/**************************************************************************************************/
/** MODELS - linear autoencoder with one hidden layer with one node **/

const size_t inputSize=trainData.n_rows;// =8 
const size_t outputSize=trainData.n_rows;// =8
const size_t hiddenLayerSize=1;//

/** could build model direct in layers - this works fine but more elegant using sequential here see below **/
/* not using this code *//*
FFN<MeanSquaredError<>,RandomInitialization> model0(MeanSquaredError<>(),RandomInitialization(-1,1));
model0.Add<Linear<> >(inputSize, hiddenLayerSize);
model0.Add<IdentityLayer<> >();
model0.Add<Linear<> >(hiddenLayerSize, outputSize);
model0.Add<IdentityLayer<> >();
*/

FFN<MeanSquaredError<>,RandomInitialization> model1(MeanSquaredError<>(),RandomInitialization(-1,1));
// Encoder
Sequential<>* encoder = new Sequential<>();
encoder->Add<Linear<> >(inputSize, hiddenLayerSize);
encoder->Add<IdentityLayer<> >();
// Decoder
Sequential<>* decoder = new Sequential<>();
decoder->Add<Linear<> >(hiddenLayerSize,outputSize);
decoder->Add<IdentityLayer<> >();

// now link all the layers together
model1.Add<IdentityLayer<> >(); // layer-0 this is needed a can't start with sequential as first layer
model1.Add(encoder);            // layer-1
model1.Add(decoder);            // layer-2

// set up optimizer 
ens::Adam opt(0.001,trainData.n_cols, 0.9, 0.999, 1e-8, 0, 1e-5,false,true); //https://ensmallen.org/docs.html#adam
                 

arma::cout<<"-------empty params------------------------"<<arma::endl;//empty params as not yet allocated
arma::cout << model1.Parameters() << arma::endl;
model1.Train(trainData, trainLabels,opt);
arma::cout<<"-------final params------------------------"<<arma::endl;
arma::cout << model1.Parameters() << arma::endl;

// Use the Predict method to get the assignments.
double loss1=model1.Evaluate(trainData,trainLabels);
arma::mat assignments;
model1.Predict(trainData, assignments);

//check forward - full forward, all layers, should be same as Predict
arma::mat assignments2;
model1.Forward(trainData, assignments2,0,2);// start layer 0, stop layer 2

// manual computation of MSE loss - from predict
double loss2=0;
for(i=0;i<assignments.n_cols;i++){
  for(j=0;j<assignments.n_rows;j++){
    loss2+= (assignments(j,i)-trainLabels(j,i))*(assignments(j,i)-trainLabels(j,i));
  }
}

// manual computation of MSE loss - from predict
double loss3=0;
for(i=0;i<assignments2.n_cols;i++){
  for(j=0;j<assignments2.n_rows;j++){
    loss3+= (assignments2(j,i)-trainLabels(j,i))*(assignments2(j,i)-trainLabels(j,i));
  }
}


// check loss manually
loss1=loss1;//get mean error from predict
loss2=loss2/ (double)assignments.n_cols;//get mean error from forward
loss3=loss3/ (double)assignments2.n_cols;//get mean error from forward
arma::cout<<"MSE auto="<<loss1<<arma::endl;
arma::cout<<"MSE manual predict="<<loss2<<arma::endl;
arma::cout<<"MSE manual forward(0,2)="<<loss3<<arma::endl;



// now compute just encoded data - push into layer 0 and out of layer 1
arma::mat assignments3;
model1.Forward(trainData, assignments3,0,1);// this works 
arma::cout<<"FORWARD SEQ 0 1 ncol"<<assignments3.n_cols<<" FORWARD SEQ 0 1 nrow"<<assignments3.n_rows<<arma::endl;

// now compute decoded data - push encoded data into layer 2 and out of layer 2
arma::mat assignments4;
model1.Forward(assignments3, assignments4,2,2);// this works but 
arma::cout<<"FORWARD 0 1 ncol"<<assignments4.n_cols<<" FORWARD 1 nrow"<<assignments4.n_rows<<arma::endl;

//now send to disk - encoded data
mat ass4T = assignments3.t();
ass4T.save("c_encoded1.csv", csv_ascii); 

//now send to disk - decoded data
mat ass5T = assignments4.t();
ass5T.save("c_decoded8.csv", csv_ascii); 



}
