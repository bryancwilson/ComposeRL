# functions for MusicAI Rewards
from music21 import *

# Based on First Species Counterpoint Guidelines
def MelodyRew(key, melodyNote, baseNotes, note_history, step, terminal):
    reward = 0

    illegal_intervals = [
        [1, 0, 0, 0, 0, 0], # minor second/major seventh
        [0, 1, 0, 0, 0, 0], # major second/minor seventh
        [1, 0, 0, 0, 0, 0], # perfect fourth or fifth
        [0, 0, 0, 0, 0, 1]  # tritone
    ]

    # Consider Melody to Bass
    if terminal:
        if chord.Chord((melodyNote, baseNotes[step])).intervalVector == [0, 0, 0, 0, 0, 0]: # make sure last choice is octave or unison
            reward+=1

    else:
        if chord.Chord((melodyNote, baseNotes[step])).intervalVector == [0, 0, 0, 0, 1, 0] and chord.Chord((note_history[-1], baseNotes[step - 1])).intervalVector == [0, 0, 0, 0, 1, 0]: # Parallel Fifths of Fourths
            reward-=1
        
        if chord.Chord((melodyNote, baseNotes[step])).intervalVector == [0, 0, 0, 0, 0, 0] and chord.Chord((note_history[-1], baseNotes[step - 1])).intervalVector == [0, 0, 0, 0, 0, 0]: # Parallel Octave or Unison
            reward-=1
        
        if chord.Chord((melodyNote, baseNotes[step])).intervalVector == [0, 0, 0, 1, 0, 0] and chord.Chord((note_history[-1], baseNotes[step - 1])).intervalVector == [0, 0, 0, 1, 0, 0]: # encourage the use of parallel thirds and sixths
            reward+=1

        if chord.Chord((melodyNote, baseNotes[step])).intervalVector in illegal_intervals: # encourage the use of parallel thirds and sixths
            reward-=1


    # Consider Melody to Previous Notes

    if terminal:
        if chord.Chord((melodyNote, note_history[-1])).intervalVector == [1, 0, 0, 0, 0, 0]: # encourage stepwise motion to final note
            reward+=1
    else:
        if step > 1: # encourage sequential contrary motion
            if pitch.Pitch(melodyNote.nameWithOctave).frequency > pitch.Pitch(note_history[-1].nameWithOctave).frequency and pitch.Pitch(note_history[-1].nameWithOctave).frequency < pitch.Pitch(note_history[-2].nameWithOctave).frequency:
                reward+=1
            elif pitch.Pitch(melodyNote.nameWithOctave).frequency < pitch.Pitch(note_history[-1].nameWithOctave).frequency and pitch.Pitch(note_history[-1].nameWithOctave).frequency > pitch.Pitch(note_history[-2].nameWithOctave).frequency:
                reward+=1
        
    return reward


