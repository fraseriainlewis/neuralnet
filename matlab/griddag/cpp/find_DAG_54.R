#------------------------------------------------------------------------------
# This file reads in the A data, then uses PC alg to find directed arcs
# and undirected arcs using pcalg. It then uses bnlearn to resolve the undirected
# arcs using hill climber.
#
# IMPORTANT NOTES: pc() might not give a valid CPDAG - this has undirected 
# and directed arcs - but can give cycles even on only the directed arcs.  
# To resolve this needs the options given in the pc() call below - u2pd=random
# this then makes sure (I think, all tests for so far seem to confirm) that there are
# no cycles in the directed arcs. All it then needs is for the undirected arcs to be resolved.
# To do this we create whitelists and blacklists and use bnlearn (it's easy in this, but
# not currently easy in abn). The method is easy but a little fiddly/tedious due to the need to 
# use variable names.
# SIDE note: a hillclimb with bnlearn can easily give 500+ arcs - super overmodelling!
#
#---------------------------------------------------------------------------------------

#----------------------------
## 0. setup 
#---------------------------
rm(list=ls());
library(pcalg);
library(bnlearn);
library(abn);
setwd("/home/l/a")
load("j.RData"); # this has lots of stuff in it
thedata<-joined_scaled; # tidy and leave only data and distributions
#rm(list=ls()[-match(c("thedata","distrib"),ls())]);
# remove repeated measures variables - categorical
thedata<-thedata[,-match(c("",""),names(thedata))]
if(ncol(thedata)!=length(distrib)){error("dimension mismatch!");}
names(distrib)<-names(thedata);# make sure names are consistent

# "thedata" - the actual data - data.frame
# "distrib" - distributions definition for abn - list

#-------------------------------------------------------------
#-------------------------------------------------------------
# 1. run PC to get directed and undirected arcs using structure contraints
#-------------------------------------------------------------
#-------------------------------------------------------------
# the u2pd="rand" option is essential - without this pc() might not return a valid CPDAG
set.seed(100);
pc.joined <- pc(suffStat=list(C = cor(thedata), n = nrow(thedata)), 
	              indepTest = gaussCItest, # conditional independence test at a certain alpha
                labels = names(thedata),
	              alpha = 0.01,
                skel.method = c("stable"),# conservative = TRUE,
                u2pd="rand"
	)   #  estimates the skeleton of the causal structure

amatM<-as(pc.joined, "amat")

if(!isValidGraph(amatM,type="cpdag")){stop("---- this must return TRUE! ----\n");
	} else {cat("ok - we have a cpdag\n");}
if(!isValidGraph(amatM,type="dag")){cat("---- need to resolve some undirected arcs ----\n");}

# grab all the edges from PC, directed and not
pc.edges<-showEdgeList(pc.joined);# undir and direct

#-------------------------------------------------------------
#-------------------------------------------------------------
# 2. check results - directed arcs - from pcalg have no cycles
#-------------------------------------------------------------
#-------------------------------------------------------------
# create empty DAG matrix with correct names
# 
dag.dir<-matrix(rep(0,ncol(thedata)^2),ncol=ncol(thedata));
colnames(dag.dir)<-rownames(dag.dir)<-names(thedata);

# fill with directed arcs
for(i in 1:nrow(pc.edges$direct)){# for each directed arc
	curset<-pc.edges$direct[i,]; # grab arc
	dag.dir[curset[2],curset[1]]<-1; # reverse row and col
}

# below fits the model with only directed arcs from pc() - this is just a check
# that there are no cycles lurking in the results from pcalg
# this line will give an error if it finds a cycle
m1<-fitabn(dag.m=dag.dir,data.df=thedata,data.dists=distrib);


#-------------------------------------------------------------
#-------------------------------------------------------------
# 3. Create white list and a black lists to search across edges to get a DAG using hillclimber
#-------------------------------------------------------------
#-------------------------------------------------------------

# create matrix for use with bnlearn which contains all the directed arcs from pcalg
# we want to keep all of these in the model
mywhite.m<-NULL;
# fill with directed arcs
for(i in 1:nrow(pc.edges$direct)){# for each directed arc from pcalg
	mywhite.m<-rbind(mywhite.m,names(thedata)[pc.edges$direct[i,]])
}

# now create a matrix with all the undirected arcs - in both directions
# we want to search within these
mymaybe.m<-NULL;
# fill with directed arcs
for(i in 1:nrow(pc.edges$undir)){# for each undirected arc
	mymaybe.m<-rbind(mymaybe.m,names(thedata)[pc.edges$undir[i,1:2]]) # one dir
	mymaybe.m<-rbind(mymaybe.m,names(thedata)[pc.edges$undir[i,2:1]]) # other dir
}

# blacklist 
# now ban all other arcs except for the undirected arcs - do this in two stages
# first ban every single arc possible, then remove from the black list the white and maybe arcs
ban.m<-NULL;
for(i in 1:length(names(thedata))){# for each variable - just a pairwise search
	for(j in 1:length(names(thedata))){
		ban.m<-rbind(ban.m,c(names(thedata)[i],names(thedata)[j]));
	}
		}

# combine whitelist and maybe list		
allkeep.m<-rbind(mywhite.m,mymaybe.m)	
# loop through the banlist and find entries in the whitelist+maybelist and remove.
# note paste() is just to make comparison easier as comparing two strings rather than arrays
dropme<-NULL; # this will stop all the rows we want to drop
for(i in 1:nrow(ban.m)){# for each banned arc
	for(j in 1:nrow(allkeep.m)){# for each arc we want unbanned
	      if(paste(ban.m[i,],collapse="")==paste(allkeep.m[j,],collapse="")){
	      	                      dropme<-c(dropme,i);
	      }
	}
}
	
# drop the arcs which we do not want to ban - those which are white or maybe
ban.m<-ban.m[-dropme,]; # drop the arcs which are either white or maybe

#-------------------------------------------------------------
#-------------------------------------------------------------
# 4. Run a DAG using hillclimber where directed arcs from pcalg are fixed
# and search space is limited to only the undirected dags
#-------------------------------------------------------------
#-------------------------------------------------------------

set.seed(100);
# now use simple hill climber with BIC to 
# this will return a final DAG
myres<-hc(thedata, whitelist = mywhite.m, blacklist=ban.m)

# now build a DAG as per abn format
for(i in 1:nrow(myres$arcs)){
	curset<-match(myres$arcs[i,],names(thedata));
	dag.dir[curset[2],curset[1]]<-1; # note reverse - as format is A->B we want B<-A
}

# finally fit with abn - this is to check for cycles as much as anything. 
m1<-fitabn(dag.m=dag.dir,data.df=thedata,data.dists=distrib);

# a final check - ideally, although not guaranteed we want a model with the same number of
# arcs as the total number of directed arcs + number of undirected arcs

print(a<-nrow(pc.edges$direct));
print(b<-nrow(pc.edges$undir));
print(d<-sum(dag.dir));
cat("ideally ",a+b," = ",d,"\n") 

# finally a plot 
graphics.off()
pdf(file="mydag.pdf",width=15,height=10)
plotabn(dag.m=dag.dir,data.dist=distrib)
dev.off()
