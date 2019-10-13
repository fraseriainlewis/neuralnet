function [InitialObservation, LoggedSignal] = myResetFunction(envConstants)
% Reset function - set to empty DAG and location top left corner
% initial state.


loc = [2];% row 2 col 1 as starting point

% Return initial environment state variables as logged signals.
LoggedSignal.State = loc;%uint32([reshape(DAG,1,4*4),loc]); % one long row vector
InitialObservation = LoggedSignal.State;

end
