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

statePath = [
    imageInputLayer([1 1 1], 'Normalization', 'none', 'Name', 'state')
    fullyConnectedLayer(24, 'Name', 'CriticStateFC1')
    leakyReluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(24, 'Name', 'CriticStateFC2')];

actionPath = [
    imageInputLayer([1 1 1], 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(24, 'Name', 'CriticActionFC1')
    leakyReluLayer('Name', 'CriticRelu2')
    fullyConnectedLayer(24, 'Name', 'CriticActionFC2')
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

criticOpts = rlRepresentationOptions('LearnRate',0.02,'GradientThreshold',1);

critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'state'},'Action',{'action'},criticOpts);


agentOpts = rlSARSAAgentOptions;
agentOpts.DiscountFactor=0.99;
agentOpts.EpsilonGreedyExploration.EpsilonMin = .05;
agentOpts.EpsilonGreedyExploration.Epsilon = .9;


agent = rlSARSAAgent(critic,agentOpts);


trainOpts = rlTrainingOptions(...
    'MaxEpisodes',100000,...
    'MaxStepsPerEpisode',150,...
    'Verbose',false,...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',11,...
    'ScoreAveragingWindowLength',30); 


doTraining = true;
if doTraining    
    % Train the agent.
    trainingStats = train(agent,env3,trainOpts);
end









