import numpy as np
import matplotlib.pyplot as plt

from music21.clef import BassClef, TrebleClef
from music21 import *

from RewardFunction import MelodyRew


class MusicComposition:
    def __init__(self, tonality, timesig, n1, n2, n3, n4, n5, n6): 
        self.PieceInit(tonality, timesig, n1, n2, n3, n4, n5, n6)
        self.stateSpace = [i for i in range(5)] # we are picking the range from A on top of the bass clef to C5 in the middle of the treble clef. There is nine ten in that range (including A and B)
        self.stateSpace.remove(4)
        self.stateSpacePlus = [i for i in range(5)]
        self.actionSpaceIndex = {'C4': 1, 'D4': 2, 'E4': 3, 'F4': 4, 'G4': 5, 'A4': 6, 'B4': 7, 'C5': 8, 'D5': 9, 'E5': 10}
        self.actionSpaceIndexRev = {1: 'C4', 2: 'D4', 3: 'E4', 4: 'F4', 5: 'G4', 6: 'A4', 7: 'B4', 8: 'C5', 9: 'D5', 10: 'E5'}
        self.actionRangeNum = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # self.actionSpace = {'A3': note.Note('A3', type='quarter', clef=TrebleClef), 'B3': note.Note('B3', type='quarter', clef=TrebleClef), 'C4': note.Note('C4', type='quarter', clef=TrebleClef), 'D4': note.Note('D4', type='quarter', clef=TrebleClef), 
        #                     'E4': note.Note('E4', type='quarter', clef=TrebleClef), 'F4': note.Note('F4', type='quarter', clef=TrebleClef), 'G4': note.Note('G4', type='quarter', clef=TrebleClef), 'A4': note.Note('A4', type='quarter', clef=TrebleClef), 
        #                     'B4': note.Note('B4', type='quarter', clef=TrebleClef), 'C5': note.Note('C5', type='quarter', clef=TrebleClef)} # This will just serve as a dictionary of actions the agent can take
        self.noteRange = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5']
        self.possibleActions = self.makeActionSpace(self.actionRangeNum)
        self.actionSpaceNum = [i for i in range(len(self.possibleActions))]
        self.actionSpaceDic = self.makeActionSpaceDict(self.possibleActions, self.actionSpaceNum)
        self.agentPosition = note.Note('C5')
        self.initPosition = note.Note('C5')
        self.currentState = 0

    def makeActionSpace(self, actionList1):
        actionSpace = []
        for action1 in actionList1:
            actionSpace.append((action1))
        return actionSpace

    def makeActionSpaceDict(self, actionSpace, actionSpaceIndex):
        actionSpaceDict = {}
        for i in range(len(actionSpaceIndex)):
            actionSpaceDict[i] = actionSpace[i]
        return actionSpaceDict


    def illegal(self, newState): #This function will give the agent a negative reward for going outside the specified pitch range. Based on the AP Testing rubric
        # if we try and move outside of the pitch range alloted to the agent
        if newState not in self.stateSpacePlus:
            return True
        else:
            return False


    def step(self, step, action, baseNotes, history): # This is the function that will allow the agent to alter its environment
        
        noteAction = note.Note(self.noteRange[action]) 
        self.currentState = step - 1

        # calculate the reward
        reward = MelodyRew(key=note.Note('C'), 
                            melodyNote=noteAction, 
                            baseNotes=baseNotes, 
                            note_history=history[:len(history) - 1], 
                            step=step,
                            terminal=self.isTerminalState(self.currentState))
    
        # determine if move is legal
        if not self.illegal(self.currentState):  
            return self.currentState, reward, \
                self.isTerminalState(self.currentState), None # This will serve as the observation, reward, whether done or note, and any other dubugging info
        else:
            return self.agentPosition, reward, \
                self.isTerminalState(self.currentState), None # This will serve as the observation, reward, whether done or note, and any other dubugging info
                
        

    def PieceInit(self, tonality, timesig, n1, n2, n3, n4, n5, n6): 
        self.initPiece = stream.Stream()

        self.baseNotes1 = note.Note(n1, type='quarter', clef=TrebleClef) #This is each note of the baseline in the bass cleff, POTENTIALLY LOOK AT A FOR LOOP FOR THIS TO ALLOW FOR HOW EVERY MANY BASENOTES THE USER WANTS TO SELECT
        self.baseNotes2 = note.Note(n2, type='quarter', clef=TrebleClef)
        self.baseNotes3 = note.Note(n3, type='quarter', clef=TrebleClef)
        self.baseNotes4 = note.Note(n4, type='quarter', clef=TrebleClef)
        self.baseNotes5 = note.Note(n5, type='quarter', clef=TrebleClef)
        self.baseNotes6 = note.Note(n6, type='quarter', clef=TrebleClef)

        self.baseNotes = [self.baseNotes1, self.baseNotes2, self.baseNotes3, self.baseNotes4, self.baseNotes5, self.baseNotes6]
        self.key = key.Key(tonality) #Input a string to dictate the key of the piece for the environment "- is a flat and # is a sharp"
        self.timesig = meter.TimeSignature(timesig) #Input a fraction for the time signature of the piece in the environment

        self.initPiece.append(self.baseNotes1) #This is the code that combines every note above in the baseline
        self.initPiece.append(self.baseNotes2)
        self.initPiece.append(self.baseNotes3)
        self.initPiece.append(self.baseNotes4)
        self.initPiece.append(self.baseNotes5)
        self.initPiece.append(self.baseNotes6)
        #self.initPiece.show() #This function will render the intitial bass clef with the baseline

    def isTerminalState(self, state): #This function returns flag if terminal state
        if state in self.stateSpacePlus and state not in self.stateSpace:
            return True
        else:
            return False

    def actionSpaceSample(self): #Returns a random action for each state; collectively a action space sample
        return np.random.choice(self.actionSpaceNum)
        # return randomAction, self.possibleActions[randomAction]
        
        

    def reset(self):
        self.agentPosition = self.initPosition
        return self.currentState

# -------------------------------------------------------------------------------------------------------        

def maxAction(Q, state, actions): #This function spits out the action with the highest value
    values = np.array([Q[state,a] for a in actions])
    action = np.argmax(values)
    return actions[action]
    
# TEST CODE
if __name__ == '__main__':
   
    env = MusicComposition('C', '4/4', 'C4', 'E4', 'F4', 'E4', 'D4', 'C4')

    # model hyperparameters
    ALPHA = 0.9 #This is the learning rate
    GAMMA = 0.99 #This is the discount factor
    EPS = .99 #e - greedy value (perecent probability it will pick the optimal state-action pair)

    Q = {}
    for state in env.stateSpacePlus:
        for action in env.actionSpaceNum:
            Q[state, action] = 0 #table of state and action pairs for our Q Learning

    numComp = 100000 #The number of episodes and time that the while loop below will be entered
    totalRewards = np.zeros(numComp) 
    for i in range(numComp): #constantly iterate through all desired episodes

        if i % 100 == 0:
            print('Starting Composition ', i) #This will print every every 100 episodes
    
        done = False 
        epRewards = 0
        observation = env.reset()

        n = 0 #initialize n
        Z = [env.initPosition] #initialize z dictionary of selected actions

        while not done: #This loop will not stop until the episode is done which is determined by the isTerminal function in the environment
            n = n + 1
            rand = np.random.random() # This returns a random number between 0 and 1 
            action = maxAction(Q, observation, env.actionSpaceNum) if rand < (1-EPS) \
                                else env.actionSpaceSample() #This is an e-greedy policy that selects all the best actions once a threshold is reached decided by epsilon. This is a way of balancing between exploration and exploitation. The more random actions (exploration) the more it COULD potentially learn better approaches.

            Z.append(note.Note(env.noteRange[action])) #This will append every index to a list for later use
            
            observation_, reward, done, info = env.step(n, action, env.baseNotes, Z) #observation in this code is the same as state
            epRewards += reward
            
            action_ = maxAction(Q, observation_, env.actionSpaceNum) #Finds the next best action in the future state in order to calculate below a more precise estimation for the current state, action pair

            Q[observation,action] = Q[observation,action] + ALPHA*(reward + \
                        GAMMA*Q[observation_,action_] - Q[observation,action]) #This function is the root of Q-Learning, This is the constant update of Q to find the most precise value for the state action pair given the state action pair after that
            observation = observation_

        '''
        if i == 750 or i == (numComp - 1):
            O = stream.Stream()
            O.append(chord.Chord([env.baseNotes1, env.initPosition]))
            O.append(chord.Chord([env.baseNotes2, env.noteRange[Z[0]]]))
            O.append(chord.Chord([env.baseNotes3, env.noteRange[Z[1]]]))
            O.append(chord.Chord([env.baseNotes4, env.noteRange[Z[2]]]))
            O.append(chord.Chord([env.baseNotes5, env.noteRange[Z[3]]]))
            O.append(chord.Chord([env.baseNotes6, env.noteRange[Z[4]]]))
            O.show()
        '''

        if EPS - 1 / numComp > 0:
            EPS -= 1 / numComp
        else:
            EPS = 0
        totalRewards[i] = epRewards
        
    '''
    O = stream.Stream()
    O.append(chord.Chord([env.baseNotes1, env.initPosition]))
    O.append(chord.Chord([env.baseNotes2, Z[0]]))
    O.append(chord.Chord([env.baseNotes3, Z[1]]))
    O.append(chord.Chord([env.baseNotes4, Z[2]]))
    O.append(chord.Chord([env.baseNotes5, Z[3]]))
    O.append(chord.Chord([env.baseNotes6, Z[4]]))
    O.show()
    '''

    average_rewards = []
    for i in range(len(totalRewards) // 100):
        average_rewards.append(np.mean(totalRewards[100*(i - 1):100*(i)]))

    plt.plot(average_rewards)
    plt.xlabel('Episodes (x100)')
    plt.ylabel('Reward')
    plt.title('Average Reward Curve')
    plt.show()
    
     

