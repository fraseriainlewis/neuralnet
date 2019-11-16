classdef DAGenv < handle
    %MYENVIRONMENT: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        n=4;
        gridmax=4; % max index on board
        loc0 = [1];% top left cell in DAG as starting point - note cells 1:16
        %dag0=zeros(n,n); % start with empty dag
        gridmap=reshape(1:16,4,4);
        dag0=reshape(1:16,4,4);
        rewardTerminal = 0; % 50 works
        reward= 0; % penalty for each step
        %bonusReward =1;
        cumReward=0;
        numsteps=0;

        %terminalState =  [0   1   1   0   0   0   1   0   0   0   0   0   0   1   1   0]; % old [5 5]
                          % this is the best DAG on the simulated data at 4x4 size
                          %  0   0   0   0
                          %  1   0   0   1
                          %  1   1   0   1
                          %  0   0   0   0
        %terminalState = [0 0 0 0;...
        %                 1 0 0 1;...
        %                 1 1 0 1;...
        %                 0 0 0 0]'; % transpose is crucial!
        terminalState = -6.47e+03/1000;
        %gridmap=reshape(1:25,5,5);

    end
    
    properties
    lookup = [];
    end

    properties
    lookupscore = [];
    end

    properties
        % Initialize system state to cell=2 (e.g. 2,1) '
        State = [];
        %disp('here also');

    end
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    %% Necessary Methods
    methods  

    

        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = DAGenv(stateLookup, scoreLookup)

            %if nargin > 0
            %this.lookup = a;
            %end
            % Initialize Observation settings
            %% this is the DAG - a matrix
            %ObservationInfo = rlFiniteSetSpec([1:(543*16)]); %rlNumericSpec([1 16]); 
            %ObservationInfo.Name = 'DAG';
            %ObservationInfo.Description = 'current DAG, n x n as 1*(n x n) vector with current location x,y appended';
            
            % Initialize Action settings   
            %ActionInfo = rlFiniteSetSpec([1 2 3 4]);
            
            %ActionInfo = rlFiniteSetSpec({[0 1],[0 0],[0 -1],... % no spatial move 
            %                  [1 1],[1 0],[1 -1],...             % left
            %                  [2 1],[2 0],[2 -1],...             % right
            %                  [3 1],[3 0],[3 -1],...             % up
            %                  [4 1],[4 0],[4 -1]})               % down
            %ActionInfo.Name = 'DAG updates';
            %ActionInfo.Description = 'DAG updates, two layer, grid move, then add/nothing/remove arc';



            
            if nargin > 0
            this.lookup = stateLookup;
            this.lookupscore = scoreLookup;
            end

            % Initialize property values and pre-compute necessary values
            %updateActionInfo(this,a);
            %disp('here')
            %disp(this.lookup(:,1))
        end
        

       %function updateActionInfo(this,a)
       %  this.lookup=a;
       % end 
       function this = set.lookup(this,stateLookup);
         this.lookup = stateLookup;
         %disp(this.lookup(:,1))
        end
        function this = set.lookupscore(this,scoreLookup);
         this.lookupscore = scoreLookup;
         %disp(this.lookupscore(:,1))
        end

        % function this = set.State(this,startState);
        % this.State = startState;
         %disp(this.lookupscore(:,1))
        %end


        % Apply system dynamics and simulates the environment with the 
        % given action for one step.
        function [Observation,Reward,IsDone,LoggedSignals] = step(this,Action)
            LoggedSignals = [];
          
            tmp=this.lookup(:,this.State); % get state from lookup
            %disp(tmp)
            % Get current DAG - reshape from lookup into square
            dag0=uint32(reshape(tmp(1:(this.n*this.n)),this.n,this.n)'); % row vector to DAG matrix
            % Get current location in DAG matrix
            
            curlocidx=tmp(this.n*this.n+1); % last entries
            % reshape into x,y grid coord
            [r,c,v]=find(this.gridmap==curlocidx);
            curloc=[r c]; % row col
            
            %disp('current dag and position, index in lookup')
            %disp(dag0)
            %disp(curloc)
            %disp(this.State)
           
            doSecondAction=1; % flag to determine whether to do add/remove action - don't do if move off the board etc
            %% execute first part of action: grid move
            if Action(1)==0 % do nothing - a non-move, but still do second action
            elseif Action(1)==1 % move left -> decrement col
                if c-1>0 % if does not take cell off the board then take action otherwise do not change
                    c=c-1;
                else doSecondAction=0; % a bad move so finish
                end
            elseif Action(1)==2 % move right -> increment col
                if c+1<=this.gridmax % if does not take cell off the board then take action otherwise do not change
                    c=c+1;
                else doSecondAction=0; % a bad move so finish
                end
            elseif Action(1)==3 % move up -> decrement row
                if r-1>0% if does not take cell off the board then take action otherwise do not change
                    r=r-1;
                else doSecondAction=0; % a bad move so finish    
                end 
            elseif Action(1)==4 % move down -> increment row
                if r+1<=this.gridmax % if does not take cell off the board then take action otherwise do not change
                    r=r+1;
                else doSecondAction=0; % a bad move so finish    
                end        
            else error(' unknown action(1)!')         
            end
            
            bonus=0;
            dag1=dag0; % copy then work on this
            %% execute second part of action: add/nothing/delete arc
            if doSecondAction 
                if Action(2)==1 % add arc if not already present 
                    if dag1(r,c)==0
                        dag1(r,c)=1; % add arc
                           %if dag1(r,c)==this.terminalState(r,c) % added an arc we need
                           %   bonus=1;
                           %end  
                    end
                elseif Action(2)==0 % do nothing

                elseif Action(2)==-1 % remove arc
                    if dag1(r,c)==1
                        dag1(r,c)=0;
                         %if dag1(r,c)==this.terminalState(r,c) % removed an arc we do not need
                         %   bonus=1;
                          %end 
                    end
                else error('unknown action(2)');
                end
          %  else disp('no action')    
            end                                 

            newloc=[r c];

            newlocidx=this.gridmap(r,c);% back to index
            
            dag1flat = [reshape(dag1',1,this.n*this.n) newlocidx];
            % the dag1' = transpose is needed because to get to the original DAGs (and hence their scores)
            % from lookup, each col in lookup needs to be to rolled back up by row (or col then transpose)
            % find which col the new dag is in in the lookup matrix
            [l,c]=ismember(dag1flat,this.lookup','rows');


            if l==false
                %disp('cannot find dag - so must have a cycle - so only do first part of action')
                %disp('reverting to old dag but new position')
                %disp(dag0)
                dag1=dag0; % revert
                dag1flat = [reshape(dag1,1,this.n*this.n) newlocidx];
                % find which col the new dag is in in the lookup matrix
                [l0,c0]=ismember(dag1flat,this.lookup','rows');
                if l0==false 
                    error('no match for dag!')
                end
                Observation = c0;
                this.reward=this.lookupscore(c0)/1000; % revert to old dag and reward is that score
                %disp('reward')
                %disp(this.reward)
                this.cumReward=this.cumReward+this.reward;
                %disp(this.cumReward)
                this.numsteps=this.numsteps+1;
            else Observation = c; % repacks dag and loc into row vector
                %disp('c')
                %disp(c)
                 this.reward=this.lookupscore(c)/1000; % reward is new dag score
                 this.cumReward=this.cumReward+this.reward;
                 this.numsteps=this.numsteps+1;
                 % disp('reward')
                %disp(this.reward)
                % disp(this.cumReward)
            end

           %disp('new dag and position, index in lookup')
           %disp(dag1)
           % disp(newloc)
           % disp(Observation)


            % Update system states
            this.State = Observation;

            %if isequal(dag1,this.terminalState)
            if this.reward>this.terminalState
                IsDone=1; % terminate as have found best DAG
                %disp('terminating')
            else 
            IsDone=0; % not yet reached terminal state
            end

            % Get reward
            if IsDone
                 Reward =  this.rewardTerminal; % found terminal state 
                 %disp('cum reward')
                 %disp(Reward)
            else Reward = this.reward;        
            end
            
           %disp('reward')
           % disp(Reward)


        end
        
        % Reset environment to initial state and output initial observation
        function InitialObservation = reset(this,startState,rv)
            
            if rv % start from random place
                initdag=dagrng(this.n);% generate random dag
                startLoc=randi([1 16],1,1);

                initdagflat = [reshape(initdag,1,this.n*this.n) startLoc]; % [.. 1] 1=top left corner
                [l0,c0]=ismember(initdagflat,this.lookup','rows');
                if ~l0 
                    error('no match for random DAG')
                end        
                InitialObservation = [c0]; % first state - this indexes cols in a matrix - 
                this.State = InitialObservation; 
                this.cumReward=0;
                this.numsteps=0; 
                %disp(InitialObservation)
            else 
                InitialObservation = [startState]; % first state - this indexes cols in a matrix - 
                this.State = InitialObservation; 
                %disp('initial state=')
                %disp(startState)
                %disp(this.State)
                tmp=this.lookup(:,this.State); % get state from lookup
                % Get current DAG - reshape from lookup into square
                dag0=uint32(reshape(tmp(1:(this.n*this.n)),this.n,this.n)'); 
                %disp('init dag')
                %disp(dag0)
                %disp('init pos')
                %disp(tmp)
                this.cumReward=0;
                this.numsteps=0; 
            end
           
        end
    end

end
