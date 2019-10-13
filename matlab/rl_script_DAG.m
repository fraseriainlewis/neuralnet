%% setup data and variables used by environment
%%

envConstants=dagEnvConstants('n4m1000.csv',30,30)

% Environmental variables - values and storage needed during the actions in the environment
% at a miminum we need these variables in the environment
%fitDAG(dag0,N,alpha_m,alpha_w,T,R)
% hasCycle=cycle(dag0,tmpDAG,tmpVec1,tmpVec2,tmpVec3);

%% this is the DAG - a matrix
ObservationInfo = rlNumericSpec([1 18]);
ObservationInfo.Name = 'DAG';
ObservationInfo.Description = 'current DAG, n x n';

%% formulate this like a gridworld with North (1), South (2), East (3), West (4) moves - on the DAG matrix, and then after each
%% location move then second part of the action is add (1), no nothing (0), remove (-1) an arc 
ActionInfo = rlFiniteSetSpec({[1 1],[1 0],[1 -1],...
                              [2 1],[2 0],[2 -1],...
                              [3 1],[3 0],[3 -1],...
                              [4 1],[4 0],[4 -1]})
ActionInfo.Name = 'DAG updates';
ActionInfo.Description = 'DAG updates, two layer, grid move, then add/nothing/remove arc';


StepHandle = @(Action,LoggedSignals) myStepFunctionDAG(Action,LoggedSignals,envConstants);
ResetHandle = @myResetFunctionDAG;

env2 = rlFunctionEnv(ObservationInfo,ActionInfo,StepHandle,ResetHandle);


%env = rlFunctionEnv(ObservationInfo,ActionInfo,'myStepFunction','myResetFunction');


rng(0);
InitialObs = reset(env2)

%[NextObs,Reward,IsDone,LoggedSignals] = step(env2,10);
