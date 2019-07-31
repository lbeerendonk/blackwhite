"""
Staircase (3-up, 1-down) for auditory detection or discrimination.
Runs on Python 2.7 because some dependencies (pygaze) do not work with Python 3.
Becomes compatible with Python 3 by changing all instances of "raw_input" to "input".

Created by Lola Beerendonk 28-05-2019

Copyright (c) 2019; All rights reserved.
"""
import numpy as np
# import psychopy
# from psychopy import prefs
# prefs.general['audioLib'] = ['sounddevice'] 

from psychopy import sound
from psychopy import core, visual, event, monitors, data, tools, misc
from psychopy.tools import monitorunittools
#import numpy as np
from numpy.random import shuffle
import copy, time
import json
import glob
import csv
import sys
import datetime
import os
import matplotlib.pyplot as plt

#sys.path.append('D:\USERS\Stijn\exptools')

from IPython import embed as shell

screenSize = [1920,1080]
number_of_trials = 2
fullscreen = True

class staircase_interleaved():
    def __init__(self,subject_initials, task): # self, parameters = {}, phase_durations=[], session=None, screen=None, tracker=None, ID=0,

        self.subject_initials = subject_initials
        self.task = task
        self.run = 0

        # Version for counterbalancing the mapping of the responses between subjects
        if int(self.subject_initials) % 2 == 1:
            self.version = 1
        elif int(self.subject_initials) % 2 == 0:
            self.version = 2

        if len(event.getKeys(keyList=['escape'])):
            print('Quit by user.')
            core.quit()

        self.setup_files()
   
        self.info={}
        self.info['startPoints']=[0.2,0.15] #initial guesses for threshold
        self.info['nTrials']=number_of_trials # number of trials per staircase per block

        screen = monitors.Monitor('testMonitor')
        screen.setSizePix(screenSize)
        screen.setWidth(52.1)
        screen.setDistance(80)
        self.win = visual.Window(size = screenSize, units='deg', monitor=screen, fullscr=fullscreen, color=(.5,.5,.5), colorSpace='rgb')

        #### Does not work in Py2.7
        # #to escape the experiment at all times
        # def escape_exp():
        #     self.win.close()
        #     core.quit()

        # event.globalKeys.clear()
        # event.globalKeys.add(key='escape',func=escape_exp)

        self.setup_stimuli()

        #----------- Run staircase first run (= run 0)
        self.run_staircase()

        # wait for participant to respond
        event.waitKeys()  

        #----------- Second run (= run 1)
        self.run = 1 

        # Use approximate thresholds (or final points) of first run as startpoints for second run
        self.info['startPoints']=[self.newstart1,self.newstart2]
         
        #self.setup_files()
        self.stairs = []
        self.feedback1.text = 'This was the second block. Press the spacebar to continue with the third and final block.'
        
        self.run_staircase()

        # event.waitKeys()

        # self.run = 2

        # # Use approximate thresholds (or final points) of first run as startpoints for second run
        # self.info['startPoints']=[self.newstart1,self.newstart2]
         
        # #self.setup_files()
        # self.stairs = []
        # self.feedback1.text = 'This is the end of the staircase procedure. \n\n\n This window will close automatically.'

        # self.run_staircase()
            
        self.feedback1.text = 'This is the end of the staircase procedure.\n\n\nThis window will close automatically.'
        self.feedback1.draw()
        self.win.flip()

        # calculate performance per tone/condition. Performance is here calculated as the percentage of correct X responses of all X responses (i.e. controlling for bias)
        if self.task == 'discrim':
            self.answered_low = self.disc_low_correct_count + self.disc_high_count - self.disc_high_correct_count
            self.answered_high = self.disc_high_correct_count + self.disc_low_count - self.disc_low_correct_count
            self.perf_low = float(self.disc_low_correct_count)/float(self.answered_low)
            self.perf_high = float(self.disc_high_correct_count)/float(self.answered_high)            
            self.disc_perf_all = (float(self.disc_low_correct_count)+float(self.disc_high_correct_count))/(float(number_of_trials)*4.0)
            print("Performance on low tone: ", self.disc_low_correct_count, "of ", self.answered_low , "correct: ", self.perf_low, "%")
            print("Performance on high tone: ", self.disc_high_correct_count, "of ", self.answered_high, "correct: ", self.perf_high, "%")
            print("Performance of run 2 and 3: ", self.disc_perf_all)
            self.perfFile.write('%.3f,%.3f,%.3f' %(self.perf_low, self.perf_high,self.disc_perf_all))

        elif self.task == 'detect':
            self.answered_absent = self.det_absent_correct_count + self.det_present_count - self.det_present_correct_count
            self.answered_present = self.det_present_correct_count + self.det_absent_count - self.det_absent_correct_count
            self.perf_absent = float(self.det_absent_correct_count)/float(self.answered_absent)
            self.perf_present = float(self.det_present_correct_count)/float(self.answered_present)
            self.det_perf_all = (float(self.det_present_correct_count)+float(self.det_absent_correct_count))/(float(number_of_trials)*4.0)
            print("Performance on absent target: ", self.det_absent_correct_count, "of ", self.answered_absent, "correct: ", self.perf_absent, "%")
            print("Performance on present target: ", self.det_present_correct_count, "of ",self.answered_present, "correct: ", self.perf_present, "%")
            print("Performance of run 2 and 3: ", self.det_perf_all)
            self.perfFile.write('%.3f,%.3f,%.3f' %(self.perf_absent, self.perf_present,self.det_perf_all))

        self.fig.savefig(self.fig_fileName + '.png')

        core.wait(3)
        self.win.close()
        #core.quit()

    def run_staircase(self):
        if self.run == 0:
            self.first_text.draw()
            self.win.flip()
            event.waitKeys('spacebar')
        #start experiment
        self.message1.draw()
        self.intro_text.draw()
        self.draw_fixation()

        self.win.flip()

        event.waitKeys() 
        self.draw_fixation()
        self.win.flip()

        # Play separate tones and noise (only before run 0)
        if self.run == 0:
            if self.task == 'detect':
                self.target_tone.draw()
            elif self.task == 'discrim':
                self.low_tone.draw()
            self.draw_fixation()
            self.win.flip()

            self.target1.setVolume(0.5)
            self.target2.setVolume(0.5)

            for i in range(0,3):
                self.target1.play()
                core.wait(0.5)
                self.target1.stop()
                core.wait(1)

            core.wait(1)
        
            if self.task == 'discrim':
                self.high_tone.draw()
                self.draw_fixation()
                self.win.flip()
                for i in range(0,3):
                    self.target2.play()
                    core.wait(0.5)
                    self.target2.stop()
                    core.wait(1)

            core.wait(1)

            self.noise_tone.draw()
            self.draw_fixation()
            self.win.flip()
            for i in range(0,3):
                self.noise.play()
                core.wait(0.5)
                self.noise.stop()
                core.wait(1)

            core.wait(1)

        # After run 0, only present tones + noise before each run.
        self.target_tone.text = 'This is the target tone with noise.'
        self.low_tone.text = 'This is the low tone with noise.'
        self.high_tone.text = 'This is the high tone with noise.'

        # Example tones so participants know what to expect.
        if self.task == 'detect':
            self.target_tone.draw()
        elif self.task == 'discrim':
            self.low_tone.draw()
        self.draw_fixation()
        self.win.flip()

        self.target1.setVolume(0.2)
        self.target2.setVolume(0.2)

        for i in range(0,3):
            self.noise.play()
            self.target1.play()
            core.wait(0.5)
            self.noise.stop()
            self.target1.stop()
            core.wait(1)

        core.wait(1)
        
        if self.task == 'discrim':
            self.high_tone.draw()
            self.draw_fixation()
            self.win.flip()
            for i in range(0,3):
                self.noise.play()
                self.target2.play()
                core.wait(0.5)
                self.noise.stop()
                self.target2.stop()
                core.wait(1)

        self.message1.draw()
        self.message2.draw()
        self.draw_fixation()
        self.win.flip()
        event.waitKeys()

        self.stairs1 = []
        self.stairs2 = []
        self.trial_counter_1 = 0
        self.trial_counter_2 = 0 #reset trial counter for this run
        self.trial_counter = 0
        self.run_order_1 = np.argsort(np.random.rand(number_of_trials)) #create new run order for each run
        self.run_order_2 = np.argsort(np.random.rand(number_of_trials))

        if self.run == 0:
            step_sizes = [2,1]
        else:
            step_sizes = 1

        for thisStart in self.info['startPoints']:
            #we need a COPY of the info for each staircase (or the changes here will be made to all the other staircases)
            thisInfo = copy.copy(self.info)
            #now add any specific info for this staircase
            thisInfo['thisStart']=thisStart 
            thisInfo['observer']='jwp' #we might want to keep track of this
            thisInfo['nTrials']=self.info['nTrials']

            thisStair = data.StairHandler(startVal=thisStart, nReversals=5, stepSizes=step_sizes, nTrials=100, nUp=1, nDown=3, extraInfo=thisInfo,
                    method='2AFC',stepType='db',minVal=0.01,maxVal=0.3)

            #applyInitialRule=True,
            self.stairs.append(thisStair)

        for trialN in range(self.info['nTrials']):
            shuffle(self.stairs)
            for thisStair in self.stairs:
                thisIntensity = next(thisStair)
                too_early = []

                if thisStair.extraInfo['thisStart'] == self.info['startPoints'][0]:
                    print('Stair 1, trial: ', self.trial_counter_1)
                    thisStaircase = 'A' #for the data file
                    self.stairs1.append(thisIntensity)
                    print('New intensity stairs 1: ',thisIntensity)
                    this_disc_stim = self.disc_stim[self.run_order_1[self.trial_counter_1]]
                    this_present = self.present[self.run_order_1[self.trial_counter_1]]
                elif thisStair.extraInfo['thisStart'] == self.info['startPoints'][1]:
                    print('Stair 2, trial: ', self.trial_counter_2)
                    thisStaircase = 'B' #for the data file
                    self.stairs2.append(thisIntensity)
                    print('New intensity stairs 2: ',thisIntensity)
                    this_disc_stim = self.disc_stim[self.run_order_2[self.trial_counter_2]]
                    this_present = self.present[self.run_order_2[self.trial_counter_2]]

                # update the difficulty (the thisIntensity)
                self.target1.setVolume(thisIntensity)
                self.target2.setVolume(thisIntensity)
                self.draw_fixation()
                self.win.flip()
                core.wait(1)

                try:
                    self.noise.play(loops=None)
                except:
                    print('Fail noise')
                    self.noise.play(loops=None)
                if self.task == 'discrim':
                    if this_disc_stim == 1:
                        try:
                            self.target1.play(loops=None)
                        except:
                            print('Fail')
                            self.target1.play(loops=None)
                    elif this_disc_stim == 2:
                        try:
                            self.target2.play(loops=None)
                        except:
                            print('Fail')
                            self.target2.play(loops=None)
                elif self.task == 'detect':
                    if this_present == 1:
                        try:
                            self.target1.play()
                        except:
                            print('Fail')
                            self.target1.play()
                core.wait(0.5)
                self.noise.stop()
                self.target1.stop()
                self.target2.stop()
                #turn red if pressed during stim
                too_early = event.getKeys(keyList=self.responses)

                if len(too_early) > 0:
                    self.draw_red_fixation()
                    self.win.flip()
                else: 
                    self.draw_fixation()
                    self.win.flip()

                # get response
                response=None
                while response==None:
                    allKeys=event.waitKeys(keyList = self.responses)
                    for thisKey in allKeys:
                        if self.task == 'discrim':
                            if self.responses[self.disc_stim_check.index(this_disc_stim)] in thisKey:
                                response = 1
                            else:
                                response = 0
                        elif self.task == 'detect':
                            if self.responses[self.present_check.index(this_present)] in thisKey:
                                response = 1
                            else:
                                response = 0
                        if thisKey in ['q', 'escape']:
                            core.quit()  # abort experiment
                        
                        # for checking performance per tone
                        if self.run > 0:        
                            if self.task == 'discrim':
                                if this_disc_stim == 1:
                                    self.disc_low_count += 1
                                    if response == 1:
                                        self.disc_low_correct_count += 1
                                elif this_disc_stim == 2:
                                    self.disc_high_count += 1
                                    if response == 1:
                                        self.disc_high_correct_count += 1
                            elif self.task == 'detect':
                                if this_present == 0:
                                    self.det_absent_count += 1
                                    if response == 1:
                                        self.det_absent_correct_count += 1
                                elif this_present == 1:
                                    self.det_present_count += 1
                                    if response == 1: 
                                        self.det_present_correct_count += 1

                self.trial_counter += 1
                if thisStair.extraInfo['thisStart'] == self.info['startPoints'][0]:
                    # self.mean_stairs1 = thisStair.mean()
                    self.trial_counter_1 += 1
                elif thisStair.extraInfo['thisStart'] == self.info['startPoints'][1]:
                    # self.mean_stairs2 = thisStair.mean()
                    self.trial_counter_2 += 1

                # inform staircase of the response
                thisStair.addResponse(response, thisIntensity)
                
                self.draw_fixation()
                self.win.flip()

                if self.task == 'discrim':
                    self.dataFile.write('%i,%s,%i,%.3f,%i\n' %(self.trial_counter, thisStaircase, this_disc_stim, thisIntensity, response))
                elif self.task == 'detect':
                    self.dataFile.write('%i,%s,%i,%.3f,%i\n' %(self.trial_counter, thisStaircase, this_present, thisIntensity, response))

                self.previous_correct = response #update
                event.clearEvents()

        if len(event.getKeys(keyList=['escape'])):
            print('Quit by user.')
            core.quit()

        self.newstart1 = float(np.mean(self.stairs1[-5:]))
        print('New start stair 1: ', self.newstart1)
        self.newstart2 = float(np.mean(self.stairs2[-5:]))
        print('New start stair 2: ', self.newstart2)

        #if there are more two or more reversals, use the average as the new startpoint
        if len(self.stairs[0].reversalIntensities) > 1: 
            self.approxThresh1 = float(np.mean(self.stairs[0].reversalIntensities[1:]))
            print('Average of ', len(self.stairs[0].reversalIntensities)-1, ' reversals: ', self.approxThresh1)
            if self.run > 0:
                self.approxThresh1_all = np.concatenate((self.approxThresh1_all,self.stairs[0].reversalIntensities[1:]),axis=None)
        else:
            self.approxThresh1 = self.newstart1
            print('Not enough reversals. Average of last five intensities: ', self.approxThresh1)
        if len(self.stairs[1].reversalIntensities) > 1: 
            self.approxThresh2 = float(np.mean(self.stairs[1].reversalIntensities[1:]))
            print('Average of ', len(self.stairs[1].reversalIntensities)-1, ' reversals: ', self.approxThresh2)
            if self.run > 0:
                self.approxThresh2_all = np.concatenate((self.approxThresh2_all,self.stairs[1].reversalIntensities[1:]),axis=None)
        else:
            self.approxThresh2 = self.newstart2
            print('Not enough reversals. Average of last five intensities: ', self.approxThresh2)
        
        self.mean_threshold_run = float(np.mean(np.array([self.approxThresh1,self.approxThresh2])))    
        print('Approximate threshold of this run: ', self.mean_threshold_run)    

        #calculate the average of the reversals (except for the first one) of all runs so far
        if len(self.approxThresh1_all) > 0 and len(self.approxThresh2_all) > 0: 
            print('Average of all reversals of stair 1 so far: ', float(np.mean(np.array(self.approxThresh1_all))))    
            print('Average of all reversals of stair 2 so far: ', float(np.mean(np.array(self.approxThresh2_all))))
            self.mean_threshold = float(np.mean(np.concatenate((self.approxThresh1_all,self.approxThresh2_all),axis=None)))
            print('Average threshold of all runs combined so far: ',self.mean_threshold)
        else:
            print('Not enough reversals to calculate a threshold. Please redo the staircase procedure.')
            self.mean_threshold = 0.0

        # Save threshold
        self.file.write('%i,%.4f,%.4f,%.4f,%f\n' %(self.run, self.approxThresh1, self.approxThresh2, self.mean_threshold_run, self.mean_threshold))

        #overwrites the file on every run so the last threshold estimate is saved
        file = open(self.fileName+'.txt','w')
        file.write(str(self.mean_threshold))
        file.close()

        xvals = range(len(self.stairs1))
        if self.run == 0:
            self.fig,self.axs = plt.subplots(nrows=3,ncols=1,sharex=True,sharey=True,figsize =(5,15))
        #self.axs[self.run].set_xticks(np.arange(0,len(self.stairs),float(len(self.stairs))/7))
        self.axs[self.run].set_yticks(np.arange(0,0.35,0.05))
        self.axs[self.run].plot(self.stairs1,'r',self.stairs2,'b')
        self.axs[self.run].set_ylabel('Stimulus intensity')
        self.axs[self.run].set_xlabel('Trial number')
        self.axs[self.run].set_title('Run ' + str(self.run))
        self.fig.savefig(self.fig_fileName + '.png') #save intermediate?

        self.feedback1.draw()
        self.draw_fixation()
        self.win.flip()

    def draw_fixation(self): #For the circle + cross fixation (see Thaler, Shutz, Goodale & Gegenfurtner (2013))
        self.fixation1.fillColor = 'black'
        self.fixation2.lineColor = 'black'
        self.fixation2.fillColor = 'black'
        self.fixation1.draw()
        self.line1.draw()
        self.line2.draw()
        self.fixation2.draw()

    def draw_red_fixation(self): #Red version of the circle + cross fixation
        self.fixation1.fillColor = 'red'
        self.fixation2.lineColor = 'red'
        self.fixation2.fillColor = 'red'
        self.fixation1.draw()
        self.line1.draw()
        self.line2.draw()
        self.fixation2.draw()

    def setup_stimuli(self): #Including some variables
        # ### For windows: 
        # self.target1 = sound.backend_sounddevice.SoundDeviceSound('A', octave=3.5, sampleRate=44100, secs=0.5, stereo=True) 
        # self.target2 = sound.backend_sounddevice.SoundDeviceSound('B', octave=3.5, sampleRate=44100, secs=0.5, stereo=True)
        # self.noise = sound.backend_sounddevice.SoundDeviceSound('TORC_424_02_h501.wav',stereo=True)

        ### For mac:
        self.target1 = sound.Sound('A', octave=3.5, sampleRate=44100, secs=0.5) 
        self.target2 = sound.Sound('B', octave=3.5, sampleRate=44100, secs=0.5)
        self.noise = sound.Sound('TORC_424_02_h501.wav')
        
        self.noise.setVolume(1)

        # For checking if responses are correct
        self.disc_stim_check = [1,2]
        self.present_check = [0,1]

        #Create equal amount of trials for each condition
        self.present = np.ones(number_of_trials,dtype=int)
        self.present[0:int(number_of_trials/2)] = 0
        self.disc_stim = np.ones(number_of_trials,dtype=int)
        self.disc_stim[0:int(number_of_trials/2)] = 2
        #print(self.disc_stim)

        if self.version == 1:
            self.responses = ['a','l','q','escape']
        elif self.version == 2:
            self.responses = ['l','a','q','escape']

        #for the improved fixation dot
        self.d1 = 0.7 #diameter outer circle. larger option: 1.5, 0.15, 7
        self.d2 = 0.05 #diameter inner circle
        self.w1 = 4 #linewidth
        self.backgroundColor = (.5,.5,.5) #Set according to the backgroundcolor of the experiment

        self.fixation1 = visual.Circle(self.win, lineColor =self.backgroundColor, lineColorSpace = 'rgb', fillColor = 'black', fillColorSpace='rgb', size=self.d1)
        self.line1 = visual.Line(self.win,lineWidth=self.w1,start=(self.d1/2,0),end=(-self.d1/2,0),lineColor=self.backgroundColor,lineColorSpace='rgb')
        self.line2 = visual.Line(self.win,lineWidth=self.w1,start=(0,self.d1/2),end=(0,-self.d1/2),lineColor=self.backgroundColor,lineColorSpace='rgb')
        self.fixation2 = visual.Circle(self.win, lineColor ='black', lineColorSpace = 'rgb', fillColor ='black', fillColorSpace='rgb', size=self.d2)
        
        self.fixation = visual.GratingStim(self.win, color=-1, colorSpace='rgb',tex=None, mask='circle', size=0.2)
        self.stairs=[]

        # create all text stimuli
        self.message1 = visual.TextStim(self.win,height=.5,pos=[0,-3],text='Hit the spacebar when you are ready.', color=(-1,-1,-1), font = 'avenir',wrapWidth=20)
        self.message2 = visual.TextStim(self.win,height=.5,pos=[0,+3],text='X',wrapWidth=30, color=(-1,-1,-1), font = 'avenir') #Empty to adapt below

        if self.task == 'discrim':
            self.first_text = visual.TextStim(self.win, pos = [0,0], wrapWidth=20, height=.5,font='avenir',color=(-1,-1,-1),text='Your task is to discriminate between two target tones embedded in noise.\n\nNow, we will estimate the appropriate difficulty for you. To do so, we will increase the difficulty until you make mistakes. Then we will decrease the difficulty again until you are performing well. This process will go on for a while in order to get a good estimate.\n\nAt some point, you will probably not hear any target anymore. Just continue, the difficulty will be adjusted.\n\nPress the spacebar to continue.')
            self.intro_text = visual.TextStim(self.win, pos=[0,+3],height=.5,text='You will now get to hear the target tones.',wrapWidth=50, color=(-1,-1,-1), font = 'avenir')
            if self.version ==1:
                self.message2.text = 'Press LEFT (green) when you hear the low tone and RIGHT (green) when you hear the high tone.'
            elif self.version==2:
                self.message2.text = 'Press LEFT (green) when you hear the high tone and RIGHT (green) when you hear the low tone.'
        
        elif self.task == 'detect':
            self.first_text = visual.TextStim(self.win, pos = [0,0], height=.5, wrapWidth=30, font='avenir',color=(-1,-1,-1),text='Your task is to detect the target tone embedded in noise.\n\nNow, we will estimate the appropriate difficulty for you. To do so, we will increase the difficulty until you make mistakes. Then we will decrease the difficulty again until you are performing well. This process will go on for a while in order to get a good estimate.\n\nAt some point, you will probably not hear the target anymore. Just continue, the difficulty will be adjusted.\n\nPress the spacebar to continue.')
            self.intro_text = visual.TextStim(self.win, pos=[0,+3],height=.5,text='You will now get to hear the target tone.',wrapWidth=20, color=(-1,-1,-1), font = 'avenir')
            if self.version==1:
                self.message2.text = 'Press LEFT (green) when the target tone is absent and RIGHT (green) when the target tone is present.'
            elif self.version==2:
                self.message2.text = 'Press LEFT (green) when the target tone is present and RIGHT (green) when the target tone is absent.'

        self.feedback1 = visual.TextStim(self.win, height=.5,pos=[0,+3], text='This was the first run. Press the spacebar to continue.',wrapWidth=20, color=(-1,-1,-1), font = 'avenir')

        self.target_tone = visual.TextStim(self.win,height=.5, pos=[0,+3], text='This is the target tone.', color=(-1,-1,-1), font = 'avenir')
        self.high_tone = visual.TextStim(self.win, height=.5,pos=[0,+3], text='This is the high tone.', color=(-1,-1,-1), font = 'avenir')
        self.low_tone = visual.TextStim(self.win, height=.5,pos=[0,+3], text='This is the low tone.', color=(-1,-1,-1), font = 'avenir')
        self.noise_tone = visual.TextStim(self.win, height=.5,pos=[0,+3], text='This is the noise sound.', color=(-1,-1,-1), font = 'avenir')

        self.newstart1 = self.newstart2 = self.approxThresh1 = self.approxThresh2 = self.mean_threshold_run = self.mean_threshold = None
        self.approxThresh1_all = []
        self.approxThresh2_all = []
        self.disc_perf_all = None
        self.det_perf_all = None

        # for collecting the performance for each condition
        self.disc_low_count = self.disc_low_correct_count = self.disc_high_count = self.disc_high_correct_count = self.det_absent_count = self.det_absent_correct_count = self.det_present_count = self.det_present_correct_count = 0

    def setup_files(self):
        if not os.path.exists('data/' + self.task + '_staircase/participant_' + self.subject_initials):
            os.makedirs('data/' + self.task + '_staircase/participant_' + self.subject_initials + '/')     

        self.fileName = os.path.join('data/' + self.task + '_staircase/participant_' + str(self.subject_initials) + '/' + str(self.subject_initials) + '_' + self.task + '_threshold')      
        self.fig_fileName = os.path.join('data/' + self.task + '_staircase/participant_' + str(self.subject_initials) + '/' + str(self.subject_initials) +'_' + self.task + '_threshold')  
        self.perf_fileName = os.path.join('data/' + self.task + '_staircase/participant_' + str(self.subject_initials) + '/' + str(self.subject_initials) +'_' + self.task + '_performance')  

        try:  # try to get a previous parameters file
            self.expInfo = fromFile('lastParams.pickle')
        except:  # if not there then use a default set
            self.expInfo = {'observer':'jwp', 'refOrientation':0}
        self.expInfo['dateStr'] = data.getDateStr()  # add the current time

        # make a text file to save data
        self.dataFile = open(self.fileName+'_trials.txt', 'w')  # a simple text file
        self.perfFile = open(self.perf_fileName+'.txt','w')
        self.file = open(self.fileName+'_runs.txt', 'w')
        self.file.write('run,th1,th2,mean_run,mean_all\n')

        if self.task == 'discrim':
            self.dataFile.write('trial,stair_nr,disc_stim,intensity,correct\n')
            self.perfFile.write('low tone,high tone,ALL\n')
        elif self.task == 'detect':
            self.dataFile.write('trial,stair_nr,present,intensity,correct\n')
            self.perfFile.write('absent,present,ALL\n')

def main(subject_initials, task):
    ts = staircase_interleaved(subject_initials = subject_initials, task = task)
    ts.run_staircase()

# if __name__ == '__main__':
#     subject_initials = raw_input("Participant: ")
#     task = raw_input("detect or discrim: ")

    #main(subject_initials=subject_initials, task = task)
