import os
from FourRooms import FourRooms
import numpy as np
import random 

#CONSTANTS
EPOCHS      = 5000
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
    Q[index][packleft][action] += l_rate*(R[index][packleft][action] + DISCOUNT*(max(Q[FR.getPosition()[0] + FR.getPosition()[1]*13][packleft])) - Q[index][packleft][action]) - visited[index][packleft][action]
    
def update_R(FR, R, prevPos, action, CellType, visited, packleft):
    
    index = prevPos[0] + prevPos[1]*13
    
    #if action hit a wall
    if prevPos == FR.getPosition():
        R[index][packleft][action] -= 2
    
    #if action hit the package
    elif (CellType == 1 or CellType == 2 or CellType == 3):
        if CellType == (4 - packleft):
            
            R[index][packleft][action] = 1000
            R[FR.getPosition()[1]*13 + FR.getPosition()[0]][packleft] = [0,0,0,0]
        else:
            R[index][packleft][action] -= 5


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
        os.system("clear")
        print("Training... number of epochs (", k, "/", EPOCHS,")")
        while not FRobj.isTerminal():
            index = prevPos[0] + prevPos[1]*13
            
            #use exploration heuristic to determine if next action is random or max action
            if random.random() < E:
                action = random.randint(0,3)
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
