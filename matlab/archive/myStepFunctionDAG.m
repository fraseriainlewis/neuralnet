function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunctionDAG(Action,LoggedSignals,envConstants)
% Custom step function to construct DAG search environment 
%
% This function applies the given action to the environment and evaluates
% the reward for one simulation step.

% 1. unpack current state - this is a row vector which has the flattened dag and current location 
n=envConstants.n;

tmp=LoggedSignals.State;
% Get current DAG
dag0=uint32(reshape(tmp(1:(n*n)),n,n)); % filled by col
% Get current location in DAG matrix
loc=tmp((n*n+1):(n*n+2)); % last two entries

% 2. determine action
% return the new dag based on the action, this might be the same dag if the action would give a cycle
dag1=updateDAG(dag0,Action, envConstants);


% 3. fit new DAG

% check for cycle	
%hasCycle=cycle(dag0,envConstants.tmpDAG,envConstants.tmpVec1,envConstants.tmpVec2,envConstants.tmpVec3);
%if (~hasCycle)
%	disp('no cycle');
%end	
%lnscore=fitDAG(dag0,envConstants.N,envConstants.alpha_m,envConstants.alpha_w,envConstants.T,envConstants.R);
%disp(lnscore)

% Transform state to observation - store in LoggedSignals and also NextObs
LoggedSignals.State = double([reshape(dag1,1,n*n) loc]); % repacks dag and loc into row vector
NextObs = LoggedSignals.State;

% Check terminal condition
% if it finds best dag reward = 10
% all other rewards are -1, including obstacles which are either moving off the 'board' or else choosing a DAG with a cycle
%X = NextObs(1);
%Theta = NextObs(3);
%IsDone = 0; %abs(X) > EnvConstants.XThreshold || abs(Theta) > EnvConstants.ThetaThresholdRadians;

if isequal(dag1,envConstants.bestDAG)
	IsDone=1; % terminate as have found best DAG
else 
	IsDone=0;

% Get reward
if isequal(dag1,envConstants.bestDAG)
    Reward = 10;
else
    Reward = -1;
end

end
