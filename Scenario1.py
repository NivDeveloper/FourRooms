from FourRooms import FourRooms
import numpy as np
import matplotlib
import random 
#random initialized state

#initialize q matrix of size 144 x 4(up, down, left, right) to all 0
#zero q value means that action cannot be taken
#each element in this table is a state action mapping

#initialize r matrix aswell
# r values store the rewards and -1 where there is a wall or out of bounds

#+- 1000 episodes for first iteration

#Exploration heuristic
# E greedy

#CONSTANTS
EPOCHS      = 10000
#LEARN_RATE  = 0.5
E_GREEDY    = 0.5   #random action if < E. max action otherwise
DEC_RATE    = 0.6   #rate at which E_GREEDY is decreased each iteration
DISCOUNT    = 0.9

#create Q table and R table with following format
#___________________
#(x,y)  | (N,S,E,W)
#-------|----------
#(0,1)  | [0,0,0,0]
#(0,2)  | [0,0,0,0]
#  .  
#  .
#  .
#(13,13)| [0,0,0,0]
#___________________
l = [0,0,0,0]
#Q table
#             N S E W  
Q = np.array([l for i in range(169)])

#R table
#             N S E W  
R = np.array([l for i in range(169)])

def update_Q(FR, Q, prevPos, action, visited):
    #index before action took place
    index = prevPos[0] + prevPos[1]*13
    l_rate = 1/(1 + 0.3*visited[index][action])
    #update Q value for position before action took place
    Q[index][action] += l_rate*(R[index][action] + DISCOUNT*(max(Q[FR.getPosition()[0] + FR.getPosition()[1]*13])) - Q[index][action])
    
    #update learning rate
    #update greedy value
    #uupdate discout value
                                                            


def update_R(FR, R, prevPos, action):
    
    index = prevPos[0] + prevPos[1]*13
    
    #if action hit a wall
    if prevPos == FR.getPosition():
        #print(prevPos, FR.getPosition(), action)
        #print("found -1")
        R[index][action] = -1
    
    #if action hit the package
    if FR.getPackagesRemaining() == 0:
        R[index][action] = 100
    


def LearningLoop(FRobj, Q, R, EPOCHS):
    """ Learning loop to choose and take action
    
        updates Q and R and chooses the action based on that
    """
    for k in range(EPOCHS):
        
        FRobj.newEpoch()
        #choose current position randomly
        prevPos = FRobj.getPosition()
        
        #table showing how many times each state action pair has been visited
        #                   N S E W  
        visited = np.array([[0,0,0,0] for x in range(169)])
        E = E_GREEDY
        #print(Q)
        while not FRobj.isTerminal():
            index = prevPos[0] + prevPos[1]*13
            #print(prevPos)
            #use exploration heuristic to determine if next action is random or max action
            #print(Q)
            if random.random() < E:
                #random action from valid actions
                choices = []
                #print("random ",prevPos)
                for i in range(4):
                    #only select from valid rewards from R
                    if R[index][i] >= 0:
                        choices.append(i)
                action = choices[random.randint(0,len(choices)-1)]
                #print(prevPos, action, choices)
                visited[index][action] += 1
                FRobj.takeAction(action)#CellType, prevPos, numpack, terminal = 
            else:
                #take max action based on max Q
                choices = []
                #print("chosen ",prevPos)
                action = 0
                for i in range(4):
                    #find list of maximum Q choices
                    if Q[index][i] == max(Q[index]):
                        #if only zeros in Q table then choose randomly from all actions
                        choices.append(i)
                        
                action = choices[random.randint(0,len(choices)-1)]
                visited[index][action] += 1
                FRobj.takeAction(action)#CellType, prevPos, numpack, terminal = 
            #print(CellType)
            update_R(FRobj, R, prevPos, action)
            update_Q(FRobj, Q, prevPos, action, visited)


            #update prevpos
            prevPos = FRobj.getPosition()
            #incrementally decrease E_GREEDY
            E *= DEC_RATE
            #update learning rate

def main():

    # Create FourRooms Object
    fourRoomsObj = FourRooms('simple')

    # This will try to draw a zero
    actSeq = [FourRooms.LEFT, FourRooms.LEFT, FourRooms.LEFT,
              FourRooms.UP, FourRooms.UP, FourRooms.UP,
              FourRooms.RIGHT, FourRooms.RIGHT, FourRooms.RIGHT,
              FourRooms.DOWN, FourRooms.DOWN, FourRooms.DOWN]

    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']

    print('Agent starts at: {0}'.format(fourRoomsObj.getPosition()))

    for act in actSeq:
        gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(act)

        print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[act], newPos, gTypes[gridType]))

        if isTerminal:
            break

    # Don't forget to call newEpoch when you start a new simulation run
    LearningLoop(fourRoomsObj, Q, R, EPOCHS)
    # Show Path
    fourRoomsObj.showPath(-1)


if __name__ == "__main__":
    main()
