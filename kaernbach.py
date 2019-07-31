# -*- coding: utf-8 -*-
"""
Kaernbach staircase setup for an auditory detection or discrimination task.
Created July 2019 by Lola Beerendonk 

Adapted from smathias
"""

import numpy as np
import psychopy
from psychopy import prefs
prefs.general['audioLib'] = ['sounddevice'] 

from psychopy import sound
from psychopy import logging, core, visual, event, monitors, data, tools, misc
from psychopy.tools import monitorunittools
from numpy.random import shuffle
import copy, time
import json
import glob
import csv
import sys
import datetime
import os
import matplotlib.pyplot as plt

logging.console.setLevel(logging.CRITICAL)

#sys.path.append('D:\USERS\Stijn\exptools')

from IPython import embed as shell

class Kaernbach1991:
    def __init__(self, subject_initials, task, dv0=.3, p=0.75, reversals=[2, 400], stepsizes=(.05, .005), initialerrfix=True, avgrevsonly=True, cap=False):
        """
        Helper class for tracking an adaptive staircase using the weighted
        transformed up/down method proposed by Kaernbach (1991). Keywords are
        used to set parameters at initialisation, but they can be changed at
        any point during the run if necessary.
        
        The main part is the method 'trial' which advances the staircase.
        Once the staircase is over, all of the data can be accessed, and
        summarised graphically using the function 'makefig' (requires
        matplotlib).
        
        stepsizes has two values (the first one larger than the second one) because it will use te first stepsize 
        until reversals[0] is reached (steep decline) and then use stepsizes[1] for all other trials. 

        """

        self.dv = dv0  #starting point contrast (in this case, volume of the tone)
        self.dvs = [] #for collecting the volume of each trial
        self.dvs4avg = [] #the average volume is only calculated from trials of phase 1
        self.p = p #percentage correct to converge on
        self.factor =  self.p / (1 - self.p) #the ratio between stepsizes up and stepsizes down. in the case of 0.75, this is 3.  
        self.reversals = reversals
        self.stepsizes = stepsizes
        self.initialerrfix = initialerrfix
        self.avgrevsonly = avgrevsonly
        self.revn = 0
        self.phase = 0
        self.staircaseover = False
        self.firsttrial = True
        self.prevcorr = None
        self.trialn = 0
        self.cap = cap

    
    def trial(self, corr):
        """
        Advance the staircase by one trial. Takes a Boolean which indicates
        whether the listener got the trial correct or incorrect.
        
        """
        # do nothing if the staircase is already over
        if not self.staircaseover: 
            # record dv if needed
            if not self.firsttrial:
                if corr != self.prevcorr:
                    reversal = True
                    self.revn += 1
                else:
                    reversal = False
            if self.phase == 1:
                if self.avgrevsonly:
                    if reversal:
                        self.dvs4avg.append(self.dv)
                else:
                    self.dvs4avg.append(self.dv)
            # initial error fix: if the dv goes above the initial value during
            # the first phase, add more reversals ...
            if self.initialerrfix:
                if not self.corr:
                    if self.trialn <= self.factor + 1:
                        self.reversals[0] += 2
                        self.initialerrfix = False

            if self.corr:
                self.dv -= (self.stepsizes[self.phase] / float(self.factor))
            else:
                self.dv += self.stepsizes[self.phase]
            # cap dv
            if self.cap:
                if self.dv > self.cap: self.dv = self.cap
            # update the object
            if self.revn >= self.reversals[0]:
                self.phase = 1
            if self.revn >= np.sum(self.reversals):
                print('nr of reversals reached')
                self.staircaseover = True
            self.firsttrial = False
            self.prevcorr = self.corr
            print(self.trialn)
            self.trialn += 1
            self.dvs.append(self.dv)
    
    def getthreshold(self):
        """
        Once the staircase is over, get the average of
        the dvs to calculate the threshold.
        
        """
        self.threshold = np.mean(self.dvs4avg)
        print('Approximate threshold: ', self.threshold)
        file = open(self.fileName+'.txt','w')
        file.write(str(self.threshold))
        file.close()

    
    def makefig(self):
        """
        View or save the staircase.
        
        """
        self.fig_fileName = os.path.join('data/' + self.task + '_staircase/participant_' + str(self.subject_initials) + '/' + str(self.subject_initials) +'_' + self.task + '_staircase')  
        x = np.arange(self.trialn) + 1   
        y = self.dvs
        plt.plot(x, y)
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))
        plt.ylabel('Dependent variable')
        plt.xlabel('Trial')
        plt.hlines(np.mean(self.dvs4avg), min(x), max(x), 'r')
        plt.savefig(self.fig_fileName + '.png')    
        #plt.show()




# def main(subject_initials,task):
#     ts = Kaernbach1991(subject_initials = subject_initials, task = task)
#     ts.run_staircase()
    #core.quit()

# if __name__ == '__main__':
#     subject_initials = raw_input("Participant: ")
#     task = raw_input("detect or discrim: ")
#     main()
