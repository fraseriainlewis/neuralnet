%% setup data and variables used by environment
%%
 cd '/Users/fraser/myrepos/neuralnet/matlab/gridworldcustom'
 clear all;
 diary myDiaryFile.txt

 env3=MyGridEnv;% template file defining class
 validateEnvironment(env3)


rng(0);
InitialObs = reset(env3)

[NextObs,Reward,IsDone,LoggedSignals] = step(env3,[2]);

obsInfo = getObservationInfo(env3);
actInfo = getActionInfo(env3);

statePath = [
    imageInputLayer([1 1 1], 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(2, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(2, 'Name', 'CriticStateFC2')];

actionPath = [
    imageInputLayer([1 1 1], 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(2, 'Name', 'CriticActionFC1')
    reluLayer('Name', 'CriticRelu2')
    fullyConnectedLayer(2, 'Name', 'CriticActionFC2')
    ];

commonPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1, 'Name', 'output')];

criticNetwork = layerGraph(statePath);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = addLayers(criticNetwork, commonPath);    
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC2','add/in2');

criticOpts = rlRepresentationOptions('LearnRate',0.05,'GradientThreshold',1);




critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'state'},'Action',{'action'},criticOpts);

agentOpts = rlQAgentOptions;
%agentOpts.UseDoubleDQN=false;   
%agentOpts.TargetUpdateMethod="periodic";
%agentOpts.TargetUpdateFrequency=4;   
%agentOpts.ExperienceBufferLength=0;
agentOpts.DiscountFactor=0.99;
agentOpts.EpsilonGreedyExploration.Epsilon = .1;
%agentOpts.MiniBatchSize=0;



Qagent = rlQAgent(critic,agentOpts);

 %'Plots','training-progress',...

trainOpts = rlTrainingOptions(...
    'MaxEpisodes', 100000, ...
    'MaxStepsPerEpisode', 20, ...
    'Verbose', false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',11,...
    'ScoreAveragingWindowLength',30);

doTraining = true;
if doTraining    
    % Train the agent.
    trainingStats = train(Qagent,env3,trainOpts);
end









