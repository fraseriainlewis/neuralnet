%run dag_setup.m;

dag0=zeros(5,5);     
dag0(1,2:3)=1
dag0(2,3:5)=1
dag0(3,4)=1
dag0(4,5)=1

fitDAG(dag0,N,alpha_m,alpha_w,T,R)
disp("should be # -8071.677")


disp("make sure to run dag_setup.m first")

n=5;
totcomb=nchoosek(5,0)+nchoosek(5,1)+nchoosek(5,2)+nchoosek(5,3)+nchoosek(5,4)+nchoosek(5,5); %combinations for one node -hard codes for n=3
a=dagTableOneNode(zeros(5,5,'uint32'),totcomb);% returns matrix of all parent combinations for a single node
b=combvec(a',a'); % combined to get all combinations for two nodes - n.b. in each col, not across rows
c=combvec(a',b); % combined to get all combinations for two nodes - n.b. in each col, not across rows
d=combvec(a',c); % combined to get all combinations for two nodes - n.b. in each col, not across rows
dagstoreflat=combvec(a',d); % finally do again and get all combinations for four nodes
% each COL in dagstoreflat is a DAG unrolled by row, we iterate through each COL, rolling up each COL back
% into a 3x3 DAG e.g. 
if false
i=12;
reshape(dagstoreflat(:,i),n,n)' % e.g. a single DAG
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
numModels=0; % keep count of number of valid DAGs = 29K
for i=1:nmodels
 curDAG=uint32(reshape(dagstoreflat(:,i),n,n)');
 hasCycle=cycle(curDAG,tmpDAG,tmpVec1,tmpVec2,tmpVec3);
 if (~hasCycle)
 	numModels=numModels+1;
 end
 end	

scores=zeros(numModels,1);
index=1;
for i=1:nmodels
 curDAG=uint32(reshape(dagstoreflat(:,i),n,n)');
 hasCycle=cycle(curDAG,tmpDAG,tmpVec1,tmpVec2,tmpVec3);
 if (~hasCycle)
 	scores(index,1)=fitDAG(curDAG,N,alpha_m,alpha_w,T,R);
 	index=index+1;
 end
 end	

plot(scores,'s','MarkerSize',1,...
    'MarkerEdgeColor','red',...
    'MarkerFaceColor',[1 .6 .6])


