% to generate all DAGs, do by orders, e.g. order a1 a2 a3 a4, then all combinations in this order
% note adjacency matrix
% 4| all subsets of 3 dim across a1 a2 a3 e.g. 0 0 0, 0 1 0, 0 1 1, 1 1 1 etc - so datTableOneNode(3,3) - nchoose(3,0:3)
% 3| all subsets of 2 dim across a1 a2 - nchoose(2, 0:2)
% 2| all subsets of 1 dim across a1 - nchoose(1,0:1)
% 1| null
% but different orders might give same DAG so this is an over-estimate

totcomb=(nchoosek(3,0)+nchoosek(3,1)+nchoosek(3,2)+nchoosek(3,3))*(nchoosek(2,0)+nchoosek(2,1)+nchoosek(2,2))*(nchoosek(1,0)+nchoosek(1,1))


a=dagTableOneNode(zeros(4,4,'uint32'),totcomb)% returns matrix of all parent combinations for a single node - 16 x 4

% see https://en.wikipedia.org/wiki/Directed_acyclic_graph for number of vertices
% choose an order 
% build all dags based on combinations of the subsets (as above), for each DAG check that we do not already have this DAG in the set, if not then add it. 