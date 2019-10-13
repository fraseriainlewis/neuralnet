function [NextObs,Reward,IsDone,LoggedSignals] = myStepFunction(Action,LoggedSignals,envConstants)
% Custom step function to construct DAG search environment 
%
% This function applies the given action to the environment and evaluates
% the reward for one simulation step.


curlocidx=LoggedSignals.State;

[r,c,v]=find(envConstants.map==curlocidx);
curloc=[r c];
newloc=curloc;
% 2. determine action and generate new move

if isequal(curloc,[2 4]) && Action ==4
	specialMove=1;
	newloc=[4 4];
	%disp('special move')
else specialMove=0;	
end

if ~specialMove
	if Action==1 % move left
	newloc(2)=newloc(2)-1; % reduce col index by 1
	elseif Action==2 % move right
    newloc(2)=newloc(2)+1; % increment col index by 1
    elseif Action==3 % move up
    newloc(1)=newloc(1)-1; % decrement row index by 1
    elseif Action==4 % move down
    newloc(1)=newloc(1)+1; % increment row index by 1
end
end

if newloc(1)>5 || newloc(1)<1 || newloc(2)>5 || newloc(2)<1
	newloc=curloc; % a move off the grid and so reset to current location as no move
end	
if isequal(newloc,[3 3]) || isequal(newloc,[3 4]) || isequal(newloc,[3 5]) || isequal(newloc,[4 3])
	newloc=curloc; % a move into a banned cell
end


%disp('new loc')
%disp(newloc)
% 3. fit new DAG

% check for cycle	
%hasCycle=cycle(dag0,envConstants.tmpDAG,envConstants.tmpVec1,envConstants.tmpVec2,envConstants.tmpVec3);
%if (~hasCycle)
%	disp('no cycle');
%end	
%lnscore=fitDAG(dag0,envConstants.N,envConstants.alpha_m,envConstants.alpha_w,envConstants.T,envConstants.R);
%disp(lnscore)

newlocidx=envConstants.map(newloc(1),newloc(2));% back to index
% Transform state to observation - store in LoggedSignals and also NextObs
LoggedSignals.State = newlocidx; % repacks dag and loc into row vector
NextObs = LoggedSignals.State;

%disp('newidx')
%disp(newlocidx)

% Check terminal condition
% if it finds best dag reward = 10
% all other rewards are -1, including obstacles which are either moving off the 'board' or else choosing a DAG with a cycle
%X = NextObs(1);
%Theta = NextObs(3);
%IsDone = 0; %abs(X) > EnvConstants.XThreshold || abs(Theta) > EnvConstants.ThetaThresholdRadians;

if isequal(newloc,envConstants.terminalState)
	IsDone=1; % terminate as have found best DAG
	%disp('terminating')
else 
	IsDone=0; % not yet reached terminal state
end

% Get reward
if IsDone
    Reward = envConstants.rewardTerminal; % found terminal state 
elseif specialMove
    Reward = envConstants.rewardSpecial;
else Reward= envConstants.reward;    
end

%disp('reward')
%disp(Reward)

end
