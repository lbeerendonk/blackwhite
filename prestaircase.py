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

import exptools
from exptools.core.trial import Trial
from exptools.core.session import EyelinkSession

from kaernbach import Kaernbach1991

logging.console.setLevel(logging.CRITICAL)

#sys.path.append('D:\USERS\Stijn\exptools')

from IPython import embed as shell

screenSize = [1920,1080]
number_of_trials = 40
fullscreen = True

class staircaseTrial(Trial):
	def __init__(self, task, parameters = {}, phase_durations=[], session=None, screen=None, tracker=None, ID=0): # self, parameters = {}, phase_durations=[], session=None, screen=None, tracker=None, ID=0,
        
        self.task = task
        self.screen = screen
        self.parameters = parameters
        self.ID = ID
        self.phase_durations = phase_durations 

        # Version for counterbalancing the mapping of the responses between subjects
        if int(initials) % 2 == 1:
            self.version = 1
        elif int(initials) % 2 == 0:
            self.version = 2

        self.session = session
        self.block = np.floor(self.ID/self.session.block_length)
        self.tracker = tracker
        self.create_stimuli()

        if self.ID == 0:
            self.session_start_time = clock.getTime()

        self.run_time = 0.0
        self.noise_played = False
        self.signal_played = False
        self.play_sound = False
        self.prestimulation_time = self.delay_1_time = self.delay_2_time = self.answer_time = self.stimulus_time = 0.0
        self.parameters.update({'answer': -1, 
                                'correct': -1,
                                'block': self.block,
                                'RT': -1
                                })
        
        self.p_v_feedback_played = self.p_a_feedback_played = self.p_stim_played = self.cue_played = self.fixLost = False

        self.stopped = False
        self.too_early = False

        self.run = 0

        #randomize trial order
        self.run_order = np.argsort(np.random.rand(number_of_trials))

        if len(event.getKeys(keyList=['escape'])):
            print('Quit by user.')
            core.quit()

        self.setup_files()

        screen = monitors.Monitor('testMonitor')
        screen.setSizePix(screenSize)
        screen.setWidth(52.1)
        screen.setDistance(80)
        self.win = visual.Window(size = screenSize, units='deg', monitor=screen, fullscr=fullscreen, color=(.5,.5,.5), colorSpace='rgb')

        self.setup_stimuli()

        super(
            DetectTrial,
            self).__init__(
            phase_durations=phase_durations,
            parameters = parameters,
            screen = self.screen,
            session = self.session,
            tracker = self.tracker          
            )

	# def setup_files(self):
 #        if not os.path.exists('data/' + self.task + '_staircase/participant_' + self.subject_initials):
 #            os.makedirs('data/' + self.task + '_staircase/participant_' + self.subject_initials + '/')     

 #        self.fileName = os.path.join('data/' + self.task + '_staircase/participant_' + str(self.subject_initials) + '/' + str(self.subject_initials) + '_' + self.task + '_threshold')         

 #        # make a text file to save data
 #        self.dataFile = open(self.fileName+'_trials.txt', 'w')  # a simple text file
        
        if self.task == 'discrim':
            self.dataFile.write('trial,disc_stim,volume,correct\n')
        elif self.task == 'detect':
            self.dataFile.write('trial,present,volume,correct\n')

    def setup_stimuli(self): #Including some variables
        ### For windows: 
        # self.target1 = sound.backend_sounddevice.SoundDeviceSound('target1.wav', stereo=True, volume=1) 
        # self.target2 = sound.backend_sounddevice.SoundDeviceSound('target2.wav', stereo=True, volume=1)
        # self.noise = sound.backend_sounddevice.SoundDeviceSound('TORC_424_02_h501.wav', stereo=True,volume=1)

        # # For checking if responses are correct
        # self.disc_stim_check = [1,2]
        # self.present_check = [0,1]

        # #Create equal amount of trials for each condition
        # self.present = np.ones(number_of_trials,dtype=int)
        # self.present[0:int(number_of_trials/2)] = 0
        # self.disc_stim = np.ones(number_of_trials,dtype=int)
        # self.disc_stim[0:int(number_of_trials/2)] = 2

        # if self.version == 1:
        #     self.responses = ['a','l','q','escape']
        # elif self.version == 2:
        #     self.responses = ['l','a','q','escape']

        # #for the improved fixation dot
        # self.d1 = 0.7 #diameter outer circle. larger option: 1.5, 0.15, 7
        # self.d2 = 0.05 #diameter inner circle
        # self.w1 = 4 #linewidth
        # self.backgroundColor = (.5,.5,.5) #Set according to the backgroundcolor of the experiment

        # self.fixation1 = visual.Circle(self.win, lineColor =self.backgroundColor, lineColorSpace = 'rgb', fillColor = 'black', fillColorSpace='rgb', size=self.d1)
        # self.line1 = visual.Line(self.win,lineWidth=self.w1,start=(self.d1/2,0),end=(-self.d1/2,0),lineColor=self.backgroundColor,lineColorSpace='rgb')
        # self.line2 = visual.Line(self.win,lineWidth=self.w1,start=(0,self.d1/2),end=(0,-self.d1/2),lineColor=self.backgroundColor,lineColorSpace='rgb')
        # self.fixation2 = visual.Circle(self.win, lineColor ='black', lineColorSpace = 'rgb', fillColor ='black', fillColorSpace='rgb', size=self.d2)
        
        # self.fixation = visual.GratingStim(self.win, color=-1, colorSpace='rgb',tex=None, mask='circle', size=0.2)
        # self.stairs=[]

        # # create all text stimuli
        # self.message1 = visual.TextStim(self.win,height=.5,pos=[0,-3],text='Hit the spacebar when you are ready.', color=(-1,-1,-1), font = 'avenir',wrapWidth=20)
        # self.message2 = visual.TextStim(self.win,height=.5,pos=[0,+3],text='X',wrapWidth=30, color=(-1,-1,-1), font = 'avenir') #Empty to adapt below

        # if self.task == 'discrim':
        #     self.first_text = visual.TextStim(self.win, pos = [0,0], wrapWidth=20, height=.5,font='avenir',color=(-1,-1,-1),text='Your task is to discriminate between two target tones embedded in noise.\n\nNow, we will estimate the appropriate difficulty for you. To do so, we will increase the difficulty until you make mistakeself. Then we will decrease the difficulty again until you are performing well. This process will go on for a while in order to get a good estimate.\n\nAt some point, you will probably not hear any target anymore. Just continue, the difficulty will be adjusted.\n\nPress the spacebar to continue.')
        #     self.intro_text = visual.TextStim(self.win, pos=[0,+3],height=.5,text='You will now get to hear the target toneself.',wrapWidth=50, color=(-1,-1,-1), font = 'avenir')
        #     if self.version ==1:
        #         self.message2.text = 'Press LEFT (green) when you hear the low tone and RIGHT (green) when you hear the high tone.'
        #     elif self.version==2:
        #         self.message2.text = 'Press LEFT (green) when you hear the high tone and RIGHT (green) when you hear the low tone.'
        
        # elif self.task == 'detect':
        #     self.first_text = visual.TextStim(self.win, pos = [0,0], height=.5, wrapWidth=30, font='avenir',color=(-1,-1,-1),text='Your task is to detect the target tone embedded in noise.\n\nNow, we will estimate the appropriate difficulty for you. To do so, we will increase the difficulty until you make mistakeself. Then we will decrease the difficulty again until you are performing well. This process will go on for a while in order to get a good estimate.\n\nAt some point, you will probably not hear the target anymore. Just continue, the difficulty will be adjusted.\n\nPress the spacebar to continue.')
        #     self.intro_text = visual.TextStim(self.win, pos=[0,+3],height=.5,text='You will now get to hear the target tone.',wrapWidth=20, color=(-1,-1,-1), font = 'avenir')
        #     if self.version==1:
        #         self.message2.text = 'Press LEFT (green) when the target tone is absent and RIGHT (green) when the target tone is present.'
        #     elif self.version==2:
        #         self.message2.text = 'Press LEFT (green) when the target tone is present and RIGHT (green) when the target tone is absent.'

        # self.feedback1 = visual.TextStim(self.win, height=.5,pos=[0,+3], text='This was the first run. Press the spacebar to continue.',wrapWidth=20, color=(-1,-1,-1), font = 'avenir')

        # self.target_tone = visual.TextStim(self.win,height=.5, pos=[0,+3], text='This is the target tone.', color=(-1,-1,-1), font = 'avenir')
        # self.high_tone = visual.TextStim(self.win, height=.5,pos=[0,+3], text='This is the high tone.', color=(-1,-1,-1), font = 'avenir')
        # self.low_tone = visual.TextStim(self.win, height=.5,pos=[0,+3], text='This is the low tone.', color=(-1,-1,-1), font = 'avenir')
        # self.noise_tone = visual.TextStim(self.win, height=.5,pos=[0,+3], text='This is the noise sound.', color=(-1,-1,-1), font = 'avenir')

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

    def run_trial(self):
        too_early = []
        self.corr = None

        print('New intensity: ',self.dv)
        this_disc_stim = self.disc_stim[self.run_order[self.trialn]]
        this_present = self.present[self.run_order[self.trialn]]

        # update the difficulty (the thisIntensity)
        self.target1 = sound.backend_sounddevice.SoundDeviceSound('target1.wav', stereo=True, volume=1) 
        self.target2 = sound.backend_sounddevice.SoundDeviceSound('target2.wav', stereo=True, volume=1)
        self.target1.setVolume(self.dv)
        self.target2.setVolume(self.dv)
        self.draw_fixation()
        self.win.flip()
        core.wait(1)
            
        if self.task == 'discrim':
            if this_disc_stim == 1:
                self.noise.play()
                self.target1.play()
            elif this_disc_stim == 2:
                self.noise.play()
                self.target2.play()
        elif self.task == 'detect':
            if this_present == 1:
                self.noise.play()
                self.target1.play()
            elif this_present == 0:
                self.noise.play()

        core.wait(0.5)
        #turn red if pressed during stim
        too_early = event.getKeys(keyList=self.responses)

        if len(too_early) > 0:
            self.draw_red_fixation()
            self.win.flip()
        else: 
            self.draw_fixation()
            self.win.flip()

        # get response
        self.corr=None
        while self.corr==None:
            allKeys=event.waitKeys(keyList = self.responses)
            for thisKey in allKeys:
                if self.task == 'discrim':
                    if self.responses[self.disc_stim_check.index(this_disc_stim)] in thisKey:
                        self.corr = 1
                    else:
                        self.corr = 0
                elif self.task == 'detect':
                    if self.responses[self.present_check.index(this_present)] in thisKey:
                        self.corr = 1
                    else:
                        self.corr = 0
                if thisKey in ['q', 'escape']:
                    core.quit()  # abort experiment

        if self.task == 'discrim':
            self.dataFile.write('%i,%i,%.3f,%i\n' %(self.trialn, this_disc_stim, self.dv, self.corr))
        elif self.task == 'detect':
            self.dataFile.write('%i,%i,%.3f,%i\n' %(self.trialn, this_present, self.dv, self.corr))

        Kaernbach1991.trial(self.corr)

    def run_staircase(self):
        for trial in range(number_of_trials):
            self.run_trial()
        
        Kaernbach1991.getthreshold()
        Kaernbach1991.makefig()
        self.win.close()
        #core.quit()

def main(subject_initials,task):
    ts = prestaircase(subject_initials = subject_initials, task = task)
    ts.run_staircase()
