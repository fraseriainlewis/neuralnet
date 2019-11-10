clear all;
 cd '/Users/fraser/myrepos/neuralnet/matlab/griddag'
if true
	run dag_setup.m
	run script_DAGtablen4.m 
end


env=DAGenv(allStates,allScores);% template file defining class



rng(1000); %0 or 1000
InitialObs = reset(env,1)

[NextObs,Reward,IsDone,LoggedSignals] = step(env,[2 1]);

[r,c]=size(allStates);% 17 row 8688 cols
numStates=c;
% create vector to hold value for each state
% initialize to zero
V=zeros(c,1); % 8688 by 1
terminal=find(allScores>=max(allScores)); % indexes of the best score - terminal condition to value function is zero
policy=randi([1 15],c,1); % each row is a state the entry is the action form the policy - 1 through 15
policyStable=zeros(c,1);
discount=0.90;


actionLookup={[0 1],[0 0],[0 -1],... % no spatial move 
                              [1 1],[1 0],[1 -1],...             % left
                              [2 1],[2 0],[2 -1],...             % right
                              [3 1],[3 0],[3 -1],...             % up
                              [4 1],[4 0],[4 -1]};
%delete 'mylog.txt'
%diary on
%diary 'mylog.txt'

% initialize environment
% first episode - run until termination or fixed number of steps? The success of the algorithm is the average number of steps 
% over episodes from start until reaches success.
for p=1:1 % for each period
	disp('new period')
	s=1; % start state
	disp('current state')
	disp(s)
	periodTotalsteps=0;
	reset(env,s); % set to initial state
	IsDone=false;% assume starting state is not the terminal state - change for random starts
i=1;
	while ~IsDone && i<15
		% take a greedy action and update V(for currentstate) 
		bestValue= -realmax;
    			for a = 1:15 % for each possible action
    				reset(env,s);% reset needed as step advances states in next line
    				[NextState,Reward,IsDone,LoggedSignals] = step(env,actionLookup{a});
    				curQ=Reward + discount*V(NextState);
    				if curQ > bestValue
    					bestValue = curQ;
    					V(s) =curQ; % update value function for just this current state
    					greedyA = a; % store best action
    					greedyNextState = NextState;% store next state from best action

    				end;	
             	end
						disp('best action =')
    					disp(greedyA)
    					disp('next state in greedy check')
    					disp(greedyNextState)
    					disp('V(s)')
    					disp(V(greedyNextState))
	%  reset to current state and step next state as per greedy action
	reset(env,s);
	[s,Reward,IsDone,LoggedSignals] = step(env,actionLookup{greedyA}); % this updates s - the current state
	disp('next state')
	disp(s)
	periodTotalsteps = periodTotalsteps + 1;
    i=i+1;
	end % end of while = period steps
disp('period=')
disp(p)
disp('number of steps needed to reach terminal')
disp(periodTotalsteps)

end % end of period loop

