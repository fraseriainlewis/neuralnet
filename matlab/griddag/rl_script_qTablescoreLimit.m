%% setup data and variables used by environment
%%
 
clear all;
 cd '/Users/fraser/myrepos/neuralnet/matlab/griddag'
if true
	run dag_setup.m
	run script_DAGtablen4.m 
end


rng(1000)
 env=MyGridEnv44DAGscoreLimit2(allStates,allScores);% template file defining class
 validateEnvironment(env)


rng(1000); %0 or 1000
InitialObs = reset(env)

[NextObs,Reward,IsDone,LoggedSignals] = step(env,[2 1]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[2 1]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[4 1]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[4 0]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[4 1]);
[NextObs,Reward,IsDone,LoggedSignals] = step(env,[1 1]);


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
trainOpts.MaxEpisodes= 100000;
trainOpts.StopTrainingCriteria = "AverageReward";
trainOpts.StopTrainingValue = 3;  % 15 =3 works with terminal reward 50
trainOpts.ScoreAveragingWindowLength = 30; % 30


doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(qAgent,env,trainOpts);
else
    % Load pretrained agent for the example.
    load('basicGWQAgent.mat','qAgent')
end

if true
	simOpts = rlSimulationOptions('MaxSteps',30);
    %% turn on step printing to see locations
    res=sim(qAgent,env,simOpts)
    res2=res.Observation.DAG.Data
    [x,y,z]=size(res2)
    mydag=reshape(allStates(1:16,res2(:,:,z)),4,4)';
    fitDAG(mydag,N,alpha_m,alpha_w,T,R)

end 





