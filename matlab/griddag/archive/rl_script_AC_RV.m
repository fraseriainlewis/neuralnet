%% setup data and variables used by environment
%%
clear all;
 cd '/Users/fraser/myrepos/neuralnet/matlab/griddag'
if true
	run dag_setup.m
	run script_DAGtablen4.m 
end

% screenshot 1 is rng(1) NN 2 middle layer, 50 steps per episode, av value 3 - start at zero DAG
% screenshot 2 is rng(100) 100 steps per episode - start at zero DAG
% screenshot 3 is rng(100) 200 steps per episode - start at zero DAG
% screenshot 4 is rng(100) 25 steps per episode - start at zero DAG
% screenshot 5 is rng(100) 25 steps per episode and RANDOM DAG - results saved in random.mat in Downloads

 env=MyGridEnv44DAGctsRV(allStates,allScores);% template file defining class
 validateEnvironment(env)


rng(1000); %0 or 1000
InitialObs = reset(env)

[NextObs,Reward,IsDone,LoggedSignals] = step(env,[7]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[7]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[13]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[14]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[13]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[4]);

obsInfo = getObservationInfo(env);
actInfo = getActionInfo(env);

criticNetwork = [
    imageInputLayer([17 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(10,'Name','CriticFCa')
    fullyConnectedLayer(10,'Name','CriticFCb')
    leakyReluLayer('Name','CriticRelu1')
    fullyConnectedLayer(1,'Name','CriticStateFC2')];

%criticNetwork = [
%   imageInputLayer([17 1 1],'Normalization','none','Name','state')
%    fullyConnectedLayer(2,'Name','CriticFCa')
%    fullyConnectedLayer(1,'Name','CriticFCb')];

criticOpts = rlRepresentationOptions('LearnRate',0.01,'GradientThreshold',1); %0.8

critic = rlRepresentation(criticNetwork,obsInfo,'Observation',{'state'},criticOpts);

%actorNetwork = [
%    imageInputLayer([17 1 1],'Normalization','none','Name','state')
%    fullyConnectedLayer(15,'Name','action')
%    softmaxLayer('Name','output')];

actorNetwork = [
    imageInputLayer([17 1 1],'Normalization','none','Name','state')
    fullyConnectedLayer(2,'Name','CriticFC')
    leakyReluLayer('Name','CriticRelu1')
    fullyConnectedLayer(15,'Name','action')
    softmaxLayer('Name','output')];

actorOpts = rlRepresentationOptions('LearnRate',0.005,'GradientThreshold',1);%if this is high nothing happens 0.08

actor = rlRepresentation(actorNetwork,obsInfo,actInfo,...
    'Observation',{'state'},'Action',{'output'},actorOpts);


agentOpts = rlACAgentOptions(...
    'NumStepsToLookAhead',25, ...
    'DiscountFactor',0.99, ...
    'SampleTime',1.0, ...
    'EntropyLossWeight',0.01); %0.01 works fine

agent = rlACAgent(actor,critic,agentOpts);


trainOpts = rlTrainingOptions(...
    'MaxEpisodes',100000,...
    'MaxStepsPerEpisode',50,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',15,... %-16 is latest screenshot
    'ScoreAveragingWindowLength',30); 


doTraining = true;
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
end


if true
	simOpts = rlSimulationOptions('MaxSteps',30);
    %% turn on step printing to see locations
    res=sim(agent,env,simOpts)
    res2=res.Observation.DAG.Data
    [x,y,z]=size(res2)
    mydag=reshape(res2(1:16,1,z),4,4);
    fitDAG(mydag,N,alpha_m,alpha_w,T,R)

end 




