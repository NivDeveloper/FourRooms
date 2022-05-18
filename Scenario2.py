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
EPOCHS      = 5000
#LEARN_RATE = 0.5
E_GREEDY    = 0.6   #random action if < E. max action otherwise
DEC_RATE    = 0.8   #rate at which E_GREEDY is decreased each iteration
DISCOUNT    = 0.4

#create Q table and R table with following format
#__________________________
#(x,y)  |numpack| (N,S,E,W)
#-------|-------|----------
#(0,1)  |   0   | [0,0,0,0]
#(0,1)  |   1   | [0,0,0,0]
#(0,1)  |   2   | [0,0,0,0]
#(0,1)  |   3   | [0,0,0,0]
#(0,2)  |   0   | [0,0,0,0]
#(0,2)  |   1   | [0,0,0,0]
#  . 
#  .
#  .
#(13,13)|   3   | [0,0,0,0]
#__________________________
p = [0,0,0,0]
l = np.array([p for i in range(4)])
#Q table
Q = np.array([np.array([[0,0,0,0] for i in range(4)]) for i in range(169)])
#R table
R = np.array([np.array([[0,0,0,0] for i in range(4)]) for i in range(169)])

def update_Q(FR, Q, prevPos, action, visited, packleft):
    #index before action took place
    index = prevPos[0] + prevPos[1]*13
    #make negative learning rate for positions that have been visited before
    
    l_rate = 1/(1 + visited[index][packleft][action])
    #update Q value for position before action took place
    Q[index][packleft][action] += l_rate*(R[index][packleft][action] + DISCOUNT*(max(Q[FR.getPosition()[0] + FR.getPosition()[1]*13][packleft])) - Q[index][packleft][action]) - 1 - visited[index][packleft][action]
    
def update_R(FR, R, prevPos, action, CellType, visited, packleft):
    
    index = prevPos[0] + prevPos[1]*13
    
    #if action hit a wall
    if prevPos == FR.getPosition():
        R[index][packleft][action] -= 2
    
    #if action hit the package
    elif CellType == 1 or CellType == 2 or CellType == 3:
        R[index][packleft][action] = 300
        R[FR.getPosition()[1]*13 + FR.getPosition()[0]][packleft] = [0,0,0,0]
    #else:
    #   R[index][packleft][action] -= 0.1*visited[index][packleft][action]


def LearningLoop(FRobj, Q, R, EPOCHS):
    """ Learning loop to choose and take action
    
        updates Q and R and chooses the action based on that
    """
    for k in range(EPOCHS):
        
        FRobj.newEpoch()
        #choose current position randomly
        prevPos = FRobj.getPosition()
        
        #table showing how many times each state action pair has been visited
        visited = np.array([np.array([[0,0,0,0] for i in range(4)]) for i in range(169)])
        #number of packages left
        E = E_GREEDY
        packleft = 3
        #print("--------------")
        print(k)
        while not FRobj.isTerminal():
            index = prevPos[0] + prevPos[1]*13
            
            #print(visited)
            #use exploration heuristic to determine if next action is random or max action
            if random.random() < E:
                #random action from valid actions
                #choices = []
                #for i in range(4):
                    #only select from valid rewards from R
                #    if R[index][packleft][i] >= 0:
                #        choices.append(i)
                action = random.randint(0,3)#choices[random.randint(0,len(choices)-1)]
                visited[index][packleft][action] += 1
                CellType, newPos, numpack, terminal =FRobj.takeAction(action)
            else:
                #take max action based on max Q
                choices = []
                action = 0
                for i in range(4):
                    #find list of maximum Q choices
                    if Q[index][packleft][i] == max(Q[index][packleft]):
                        #if only zeros in Q table then choose randomly from all actions
                        choices.append(i)
                        
                action = choices[random.randint(0,len(choices)-1)]
                #print(Q[index], action)
                visited[index][packleft][action] += 1
                CellType, newPos, numpack, terminal = FRobj.takeAction(action)
            update_R(FRobj, R, prevPos, action, CellType, visited, packleft)
            update_Q(FRobj, Q, prevPos, action, visited, packleft)


            #update prevpos
            prevPos = FRobj.getPosition()
            #update packleftt and incrementally decrease E_GREEDY
            if CellType > 0:
                #E = E_GREEDY
                packleft -= 1
            else:
                E *= DEC_RATE
            #update learning rate

def main():

    # Create FourRooms Object
    fourRoomsObj = FourRooms('multi')

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
