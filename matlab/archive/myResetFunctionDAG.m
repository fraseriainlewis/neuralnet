function [InitialObservation, LoggedSignal] = myResetFunctionDAG(envConstants)
% Reset function - set to empty DAG and location top left corner
% initial state.

n=envConstants.n;
dag0=zeros(n,n); % start with empty dag
loc = [1,1];% top left cell in DAG as starting point

% Return initial environment state variables as logged signals.
LoggedSignal.State = [reshape(dag0,1,n*n) loc];%uint32([reshape(DAG,1,4*4),loc]); % one long row vector
InitialObservation = LoggedSignal.State;

end
