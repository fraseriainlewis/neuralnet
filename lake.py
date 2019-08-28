import gym
import numpy as np
import tensorflow as tf

from numpy.random import seed # numpy random number set function

np.random.seed(9990)# 9990

env = gym.make('FrozenLake-v0')

tf.compat.v1.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.compat.v1.placeholder(shape=[1,16],dtype=tf.float32)
W = tf.Variable(tf.random.uniform([16,4],0,0.01)) # shape, minval, maxval
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.compat.v1.placeholder(shape=[1,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.initialize_all_variables()

# Set learning parameters
y = .99
e = 0.1
num_episodes = 2000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
with tf.compat.v1.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        print("episode=",i)
        rAll = 0
        d = False
        j = 0
        #The Q-Network
        while j < 99:
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            # a = index in allQ which has the max value - the index of the action to take
            # we pass the current state vector to the network and it computes Q(s,a1), Q(s,a2),..Q(s,an)
            # now sess.run build the computational graph
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(16)[s:s+1]})
            ##print("a=",a)
            ##print("length a=",a.shape)
            ##print("allQ=",allQ)
            ##print("allQ shape=",allQ.shape)
            # epsilon-greedy - change action with chance e
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()
            #Get new state and reward from environment for taking action a
            s1,r,d,_ = env.step(a[0])
            ##print("s1=",s1)
            ##print("r=",r)
            ##print("d=",d)
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(16)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            ##print("Q1=",Q1)
            maxQ1 = np.max(Q1)
            ##print("maxQ1=",maxQ1)
            targetQ = allQ
            ##print("targetQ=",targetQ)
            ##print("targetQ shape=",targetQ.shape)
            # update the original Q with the new estimate based on subsequent reward (r) and some discounting (y)
            targetQ[0,a[0]] = r + y*maxQ1
            ##print("targetQ[0,a[0]], a[0]=",targetQ[0,a[0]]," ",a[0])
            #Train our network using target and predicted Q values
            ##print("nextQ Qout=",targetQ," ",Q1)
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            # weights are in W1
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                ##print("breaking...reached ",j,"steps\n")
                break
        jList.append(j)
        rList.append(rAll)
print("jList=",jList) 
#print('{10:.5f}'.format(sum(rList)))
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

