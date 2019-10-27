%% setup data and variables used by environment
%%
 env=MyGridEnv44DAG(allStates);% template file defining class
 validateEnvironment(env)


rng(0);
InitialObs = reset(env)

[NextObs,Reward,IsDone,LoggedSignals] = step(env,[0 0]);


qTable = rlTable(getObservationInfo(env),getActionInfo(env));
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
trainOpts.MaxEpisodes= 2000;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 19;
trainOpts.ScoreAveragingWindowLength = 30;


doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(qAgent,env,trainOpts);
else
    % Load pretrained agent for the example.
    load('basicGWQAgent.mat','qAgent')
end

simOpts = rlSimulationOptions('MaxSteps',20);
%% turn on step printing to see locations
res=sim(qAgent,env,simOpts)







