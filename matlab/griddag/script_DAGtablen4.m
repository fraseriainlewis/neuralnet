%run dag_setup.m;

dag0=zeros(4,4);     
dag0(1,2:3)=1;
dag0(2,3:4)=1;
dag0(3,4)=1;

fitDAG(dag0,N,alpha_m,alpha_w,T,R)
disp("should be -6616.118")


disp("make sure to run dag_setup.m first")

n=4;
totcomb=nchoosek(4,0)+nchoosek(4,1)+nchoosek(4,2)+nchoosek(4,3)+nchoosek(4,4); %combinations for one node -hard codes for n=3
a=dagTableOneNode(zeros(4,4,'uint32'),totcomb);% returns matrix of all parent combinations for a single node - 16 x 4
[r,c]=size(a);
dagstoreflat=zeros(r^4,4*4);
index=1;
for i=1:r
    for j=1:r
        for k=1:r
            for l=1:r
            dagstoreflat(index,:)=[a(i,:) a(j,:) a(k,:) a(l,:)];
            index=index+1;
            end
        end
    end
end

dagstoreflat=dagstoreflat';

%b=combvec(a',a'); % combined to get all combinations for two nodes - n.b. in each col, not across rows
%c=combvec(a',b); % combined to get all combinations for two nodes - n.b. in each col, not across rows
% each COLUMN is a DAG
%dagstoreflat=combvec(a',c); % finally do again and get all combinations for four nodes


% IMPORTANT NOTE: each col in dagstoreflat needs to be rebuilt into a DAGs ROWWISE, not COLWISE, which is what reshape does
% so after reshape must also do transpose to get the correct DAG defintion

% each COL in dagstoreflat is a DAG unrolled by row, we iterate through each COL, rolling up each COL back
% into a 3x3 DAG e.g. 
if true
i=12;
disp(reshape(dagstoreflat(:,i),n,n)') % e.g. a single DAG
end

%ans =
%
%     0     0     1
%     1     0     0
%     0     0     0

[tmp,nmodels]=size(dagstoreflat);
tmpDAG=zeros(n,n,'uint32');
tmpVec1=zeros(1,n,'uint32');
tmpVec2=zeros(1,n,'uint32');
tmpVec3=zeros(1,n,'uint32');
% so now loop through each col in dagstore, build a DAG, check for a cycle - do this to get the total count, no score computation yet
numValidModels=0; % keep count of number of valid DAGs
goodDAGS=zeros(1,nmodels);% will contain a 1 if no cycle and 0 if cycle
for i=1:nmodels
 curDAG=uint32(reshape(dagstoreflat(:,i),n,n)');% note transpose
 hasCycle=cycle(curDAG,tmpDAG,tmpVec1,tmpVec2,tmpVec3);
 if (~hasCycle)
 	numValidModels=numValidModels+1;
 	goodDAGS(1,i)=1;
 end
 end	
disp('total no. of DAGs=')
disp(numValidModels)

idx=find(goodDAGS>0);% this is the array of indexes of good DAGS, i.e. those without cycles



scores=zeros(numValidModels,1);
index=1;
for i=1:numValidModels % for each index in idx, i.e. for each DAG which does not have a cycle compute the score
    curDAG=uint32(reshape(dagstoreflat(:,idx(i)),n,n)');
 	scores(index,1)=fitDAG(curDAG,N,alpha_m,alpha_w,T,R);
 	index=index+1;
 end	

 % get best DAG 
bestScoreIDX=find(scores==max(scores));
bestDAG44=uint32(reshape(dagstoreflat(:,idx(bestScoreIDX)),n,n)')
bestDAG44flat=dagstoreflat(:,idx(bestScoreIDX));
fitDAG(bestDAG44,N,alpha_m,alpha_w,T,R)

 %  0   0   0   0
 %  1   0   0   1
 %  1   1   0   1
 %  0   0   0   0

% TO-DO - below is the array of all 543 DAGs but need to 
dagstoreflatDAGS=dagstoreflat(:,idx); % this contains flattened DAGS - ie. no cycles, each col
scoresDAG=scores'; % scores match cols of dagstoreflatDAGS


[a,b]=ismember(bestDAG44flat',dagstoreflatDAGS','rows')

allStates=[]
allScores=[]
for i=1:16
	tmp=[dagstoreflatDAGS; ones(1,543)*i];
	allStates=[allStates tmp];
	allScores=[allScores scoresDAG];
end	

%so all states has last row = 1 or 2 or... up to 16 - grid index flattened - blocks of 543 (each DAG a col)



%plot(scores,'-s','MarkerSize',10,...
%    'MarkerEdgeColor','red',...
%    'MarkerFaceColor',[1 .6 .6])


