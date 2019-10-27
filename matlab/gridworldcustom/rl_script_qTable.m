%% setup data and variables used by environment
%%
 env3=MyGridEnv;% template file defining class
 validateEnvironment(env3)


rng(0);
InitialObs = reset(env3)

[NextObs,Reward,IsDone,LoggedSignals] = step(env3,[2 0]);


qTable = rlTable(getObservationInfo(env3),getActionInfo(env3));
tableRep = rlRepresentation(qTable);
tableRep.Options.LearnRate = 1;

agentOpts = rlQAgentOptions;
agentOpts.EpsilonGreedyExploration.Epsilon = .04;
qAgent = rlQAgent(tableRep,agentOpts);


tableRep = rlRepresentation(qTable);
tableRep.Options.LearnRate = 1;

agentOpts = rlQAgentOptions;
agentOpts.EpsilonGreedyExploration.Epsilon = .04;
qAgent = rlQAgent(tableRep,agentOpts);

trainOpts = rlTrainingOptions;
trainOpts.MaxStepsPerEpisode = 50;
trainOpts.MaxEpisodes= 200;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 11.5;
trainOpts.ScoreAveragingWindowLength = 30;


doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(qAgent,env3,trainOpts);
else
    % Load pretrained agent for the example.
    load('basicGWQAgent.mat','qAgent')
end

%% turn on step printing to see locations
sim(qAgent,env3)







