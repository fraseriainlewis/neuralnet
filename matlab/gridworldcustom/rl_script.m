%% setup data and variables used by environment
%%

envConstants=setEnvConstants()

% Environmental variables - values and storage needed during the actions in the environment
% at a miminum we need these variables in the environment
%fitDAG(dag0,N,alpha_m,alpha_w,T,R)
% hasCycle=cycle(dag0,tmpDAG,tmpVec1,tmpVec2,tmpVec3);

%% this is the DAG - a matrix
ObservationInfo = rlFiniteSetSpec([1:25]);
ObservationInfo.Name = 'DAG';
ObservationInfo.Description = 'current DAG, n x n';

%% formulate this like a gridworld with North (1), South (2), East (3), West (4) moves - on the DAG matrix, and then after each
%% location move then second part of the action is add (1), no nothing (0), remove (-1) an arc 
ActionInfo = rlFiniteSetSpec([1 2 3 4])
%ActionInfo.Name = 'DAG updates';
%ActionInfo.Description = 'DAG updates, two layer, grid move, then add/nothing/remove arc';


StepHandle = @(Action,LoggedSignals) myStepFunction(Action,LoggedSignals,envConstants);
ResetHandle = @() myResetFunction(envConstants);

env2 = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);


%env = rlFunctionEnv(ObservationInfo,ActionInfo,'myStepFunction','myResetFunction');


rng(0);
InitialObs = reset(env2)

[NextObs,Reward,IsDone,LoggedSignals] = step(env2,[2]);



qTable = rlTable(getObservationInfo(env2),getActionInfo(env2));
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
trainOpts.StopTrainingValue = 11;
trainOpts.ScoreAveragingWindowLength = 30;


doTraining = true;

if doTraining
    % Train the agent.
    trainingStats = train(qAgent,env2,trainOpts);
else
    % Load pretrained agent for the example.
    load('basicGWQAgent.mat','qAgent')
end

plot(env2)
env.Model.Viewer.ShowTrace = true;
env.Model.Viewer.clearTrace;







