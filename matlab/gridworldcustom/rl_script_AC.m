%% setup data and variables used by environment
%%
%% good - reward 3 = critic 0.02, actor 0.08, lookahead 25, discount 0.99, entropy 0.7, maxstepsepisode 50 - one node NN
%% good - reward 3 = critic 0.04, actor 0.08, lookahead 25, discount 0.99, entropy 0.7, maxstepsepisode 50 - 2 node hidden layer
%% bad low green   critic 0.1
 cd '/Users/fraser/myrepos/neuralnet/matlab/gridworldcustom'
 clear all;
 diary myDiaryFile.txt

 env3=MyGridEnv;% template file defining class
 validateEnvironment(env3)


rng(1);
InitialObs = reset(env3)

[NextObs,Reward,IsDone,LoggedSignals] = step(env3,[2]);

obsInfo = getObservationInfo(env3);
actInfo = getActionInfo(env3);

criticNetwork = [
    imageInputLayer([1 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(10,'Name','CriticFC')
    fullyConnectedLayer(10,'Name','CriticFCa')
    leakyReluLayer('Name','CriticRelu1')
    fullyConnectedLayer(1,'Name','CriticStateFC2')];

criticOpts = rlRepresentationOptions('LearnRate',0.01,'GradientThreshold',1); %0.8

critic = rlRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);

actorNetwork = [
    imageInputLayer([1 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(2,'Name','CriticFC')
    leakyReluLayer('Name','CriticRelu1')
    fullyConnectedLayer(4,'Name','action')
    softmaxLayer('Name','output')];

actorOpts = rlRepresentationOptions('LearnRate',0.005,'GradientThreshold',1);%if this is high nothing happens 0.08

actor = rlRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'output'},actorOpts);


agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',25, ...
    'DiscountFactor',0.99, ...
    'SampleTime',1.0, ...
    'EntropyLossWeight',0.1); %0.5

agent = rlACAgent(actor,critic,agentOpts);


trainOpts = rlTrainingOptions(...
    'MaxEpisodes',50000,...
    'MaxStepsPerEpisode',250,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',31,...
    'ScoreAveragingWindowLength',30); 


doTraining = true;
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env3,trainOpts);
end








