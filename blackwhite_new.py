""""
Auditory detection and discrimination task

Created by Lola Beerendonk, 2019

Copyright (c) 2018; All rights reserved.
"""
import numpy as np
from numpy import random
import sounddevice, soundfile
import psychopy
from psychopy import prefs
prefs.general['audioLib'] = ['sounddevice']
prefs.general['audioDriver'] = ['portaudio']
#print(prefs)
from psychopy import sound
import pygame
import os, sys, datetime
import subprocess
import datetime, time
import pickle as pkl
import pandas as pd
import json
from math import *
from IPython import embed as shell
import shutil
from psychopy import logging, visual, clock, sound, event, data, core, monitors
from psychopy import parallel
from psychopy.sound import Sound

logging.console.setLevel(logging.CRITICAL)

sys.path.append('/Users/lolabeerendonk/Documents/reps/exptools')
#sys.path.append('D:\USERS\Stijn\exptools')

import exptools
from exptools.core.trial import Trial
from exptools.core.session import EyelinkSession

from psychopy.tools.attributetools import attributeSetter, setAttribute
from psychopy.visual import GratingStim, TextStim, ImageStim, DotStim, Window

# The following imports are copied from simple_tracker_experiment.py
import constants
import pygaze
from pygaze import libscreen
from pygaze import libtime
from pygaze import liblog
from pygaze import libinput
from pygaze import eyetracker

p = ['FA', 'MISS']

fullscr = False
tracker_on = False
use_parallel = False

total_trials = 40
miniblocks = 2
NR_TRIALS = total_trials/miniblocks
block_length = 10

nr_staircase_trials = 10
# trials_per_block = 85
# conditions = 4
# trials_per_condition = 255
# total_number_trials = float(trials_per_block * conditions * trials_per_condition / trials_per_block)
# if total_number_trials % 1 != 0:
#     raise ValueError('The requested amount of trials does not add up to a round number.')

stim_size = 17
noise_size = 19

if use_parallel:
    from ctypes import windll
    portaddress = 0xA0D0
    port = windll.inpout32
    
class DetectTrial(Trial):
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
        self.parameters.update({'confidence':-1,
                                'answer': -1, 
                                'correct': -1,
                                'block': self.block,
                                'RT': -1
                                })
        
        self.p_v_feedback_played = self.p_a_feedback_played = self.p_stim_played = self.cue_played = self.fixLost = False

        self.stopped = False
        self.too_early = False

        super(
            DetectTrial,
            self).__init__(
            phase_durations=phase_durations,
            parameters = parameters,
            screen = self.screen,
            session = self.session,
            tracker = self.tracker          
            )

    def create_stimuli(self):
        self.target1 = sound.backend_sounddevice.SoundDeviceSound('target1.wav', stereo=True,volume=1)
        self.target2 = sound.backend_sounddevice.SoundDeviceSound('target2.wav', stereo=True,volume=1)
        self.noise = sound.backend_sounddevice.SoundDeviceSound('TORC_424_02_h501.wav', stereo=True,volume=1.0)
        self.center = ( self.screen.size[0]/2.0, self.screen.size[1]/2.0 )



        intro_text = None

        if self.session.background == 'staircase':
            if self.task == 'detect':
                part1 = """Your task is to detect the target tone embedded in noise.
First, the appropriate difficulty level will be estimated. To do so, the difficulty will be increased until you make mistakes. Then the difficulty will be decreased again until you are performing well. This process will go on for a while in order to get a good estimate.
At some point, you will probably not hear the target anymore. Just continue, the difficulty will be adjusted.\n\n"""
                if self.version ==1:
                    part2 = """Press LEFT (green) when the target tone is absent and RIGHT (green) when the target tone is present.

Press the spacebar to continue."""
                elif self.version==2:
                    part2 = """Press LEFT (green) when the target tone is present and RIGHT (green) when the target tone is absent.

Press the spacebar to continue."""
                
            elif self.task == 'discrim':
                part1 = """Your task is to discriminate between two target tones embedded in noise.
First, the appropriate difficulty level will be estimated. To do so, the difficulty will be increased until you make mistakes. Then the difficulty will be decreased again until you are performing well. This process will go on for a while in order to get a good estimate.
At some point, you will probably not hear the target anymore. Just continue, the difficulty will be adjusted.\n\n""" 
                if self.version ==1:
                    part2 = """Press LEFT (green) when you hear the low tone and RIGHT (green) when you hear the high tone.

Press the spacebar to continue."""
                elif self.version==2:
                    part2 = """Press LEFT (green) when you hear the high tone and RIGHT (green) when you hear the low tone.

Press the spacebar to continue."""  
            intro_text = part1 + part2    
            self.message2 = visual.TextStim(self.screen, font='avenir', pos=[0,0],text=part2,color=(-1,-1,-1), wrapWidth=25) 

        elif self.session.background != 'staircase':
            if self.task == 'detect':
                if self.version == 1:
                    intro_text = """Your task is to detect the target tone embedded in noise.
Do you hear the target tone? And how confident are you?
Please indicate your answer using these buttons:

        A                      S                            K                          L
no and sure     no and unsure      yes and unsure     yes and sure

Try to answer when the stimulus is not audible anymore. If you are too fast or too slow (1.5 seconds) the fixation dot will turn red.

Press the spacebar to start."""   
                elif self.version == 2: 
                    intro_text = """Your task is to detect the target tone embedded in noise.
Do you hear the target tone? And how confident are you?
Please indicate your answer using these buttons:

        A                        S                            K                        L
yes and sure     yes and unsure      no and unsure     no and sure

Try to answer when the stimulus is not audible anymore. If you are too fast or too slow (1.5 seconds) the fixation dot will turn red.

Press the spacebar to start."""
            elif self.task == 'discrim':
                if self.version == 1:
                    intro_text = """Your task is to discriminate between two target tones embedded in noise.
Do you hear the high or the low tone? And how confident are you?
Please indicate your answer using these buttons:

        A                        S                              K                            L
low and sure     low and unsure      high and unsure     high and sure

Try to answer when the stimulus is not audible anymore. If you are too fast or too slow (1.5 seconds) the fixation dot will turn red.

Press the spacebar to start."""
                elif self.version == 2:
                    intro_text = """Your task is to discriminate between two target tones embedded in noise.
Do you hear the high or the low tone? And how confident are you?
Please indicate your answer using these buttons:

         A                         S                             K                           L
high and sure     high and unsure      low and unsure     low and sure

Try to answer when the stimulus is not audible anymore. If you are too fast or too slow (1.5 seconds) the fixation dot will turn red.

Press the spacebar to start."""

         
        if self.ID % self.session.block_length == 0 and self.ID > 0:
            #perf = np.array(self.session.corrects)[-self.session.block_length:][np.array(self.session.corrects)[-self.session.block_length:] >= 0].sum() / float(self.session.block_length) * 100.0
            early = len(np.array(self.session.corrects)[-self.session.block_length:][np.array(self.session.corrects)[-self.session.block_length:] == -2]) / float(self.session.block_length) * 100.0
            misses = len(np.array(self.session.corrects)[-self.session.block_length:][np.array(self.session.corrects)[-self.session.block_length:] == -1]) / float(self.session.block_length) * 100.0
            conf = np.array(self.session.confidence)[-self.session.block_length:][np.array(self.session.confidence)[-self.session.block_length:] >= 0].sum() / float(self.session.block_length) * 100.0

            if conf<40 or conf>60:
                feedback_text = """You were more confident in %i%% of the trials. Please try to keep this to 50%%.\n\nYou responded too quickly to %i%% of trials and you missed %i%% of trials. Please try to keep this to a minimum.\n\nPress the spacebar to continue.""" % (conf,early,misses)     
            else:
                feedback_text = """You were more confident in %i%% of the trials.\n\nYou responded too quickly to %i%% of trials and you missed %i%% of trials. Please try to keep this to a minimum.\n\nPress the spacebar to continue.""" % (conf,early,misses)    

            print(str(conf) + ' percent high conf')
            print(str(early) + ' percent fast response trials')
            print(str(misses) + ' percent missed trials')
        else: 
            feedback_text = None

        if self.session.background == 'staircase':
            textcolor = (-1,-1,-1)
            self.firstscreen = visual.TextStim(self.screen, font='avenir', pos=[0,0], text='Thank you for participating in this experiment.\n\nToday you will be performing auditory detection and discrimination tasks againts bright and dark backgrounds. Throughout the experiment, you will be listening to tones embedded in noise. You will now get to hear the target tone(s) and the noise.\n\nPress the spacebar to continue.',color=textcolor,wrapWidth=40)
        else:
            textcolor = (0,0,0)
            if self.task == 'detect' and self.ID==0:
                self.firstscreen = visual.TextStim(self.screen, font='avenir', pos=[0,0], text='You will now perform the DETECTION task.\nYou will now get to hear the target tone embedded in noise.\n\nPress the spacebar to continue.',color=textcolor,wrapWidth=40)
            elif self.task == 'detect':
                self.firstscreen = visual.TextStim(self.screen,font='avenir', pos=[0,0], text='You will now get to hear the target tone again so you know what to look for.\n\nPress the spacebar to continue.',color = (0,0,0),wrapWidth=40) 
            elif self.task == 'discrim' and self.ID==0:
                self.firstscreen = visual.TextStim(self.screen, font='avenir', pos=[0,0], text='You will now perform the DISCRIMINATION task.\nYou will now get to hear the target tone embedded in noise.\n\nPress the spacebar to continue.',color=textcolor,wrapWidth=40)
            elif self.task == 'discrim':
                self.firstscreen = visual.TextStim(self.screen, font='avenir', pos=[0,0], text='You will now get to hear the target tones again so you know what to look for.\n\nPress the spacebar to continue.',color = (0,0,0),wrapWidth=40) 

        if self.ID > 0:
            self.feedback = visual.TextStim(self.screen, font='avenir', pos=[0,0],text=feedback_text, color =textcolor,wrapWidth=40)                           
        self.message = visual.TextStim(self.screen, font='avenir', pos=[0,0],text=intro_text,color=textcolor,wrapWidth=40)

        self.target_tone_first = visual.TextStim(self.screen, pos=[0,+3], text='This is the target tone.', color=textcolor, font = 'avenir')
        self.high_tone_first = visual.TextStim(self.screen,pos=[0,+3], text='This is the high tone.', color=textcolor, font = 'avenir')
        self.low_tone_first = visual.TextStim(self.screen,pos=[0,+3], text='This is the low tone.', color=textcolor, font = 'avenir')
        self.noise_tone_first = visual.TextStim(self.screen,pos=[0,+3], text='This is the noise sound.', color=textcolor, font = 'avenir')

        self.target_tone = visual.TextStim(self.screen, font='avenir', pos=[0,+3], text='This is the target tone with noise.',color = textcolor)
        self.high_tone = visual.TextStim(self.screen, font='avenir', pos=[0,+3], text='This is the high tone with noise.',color = textcolor)
        self.low_tone = visual.TextStim(self.screen, font='avenir', pos=[0,+3], text='This is the low tone with noise.',color = textcolor)
 
        self.d1 = 1.1 #diameter outer circle. larger option: 1.5, 0.15, 7 (0.7, 0.05, 4)
        self.d2 = 0.1 #diameter inner circle
        self.w1 = 5 #linewidth
        self.backgroundColor = (0,0,0) #Set according to the backgroundcolor of the experiment
        
        if self.session.background != 'staircase':
            self.fixation_color = (0,0,0)
        else:
            self.fixation_color = (-1,-1,-1)

        self.fixation1 = visual.Circle(self.screen, lineColor=self.backgroundColor, lineColorSpace = 'rgb', fillColor = self.fixation_color, fillColorSpace='rgb', size=self.d1, units='deg')
        self.line1 = visual.Line(self.screen,lineWidth=self.w1,start=(self.d1/2,0),end=(-self.d1/2,0),lineColor=self.backgroundColor,lineColorSpace='rgb')
        self.line2 = visual.Line(self.screen,lineWidth=self.w1,start=(0,self.d1/2),end=(0,-self.d1/2),lineColor=self.backgroundColor,lineColorSpace='rgb')
        self.fixation2 = visual.Circle(self.screen, lineColor=self.fixation_color, lineColorSpace = 'rgb', fillColor =self.fixation_color, fillColorSpace='rgb', size=self.d2)

    def draw_fixation(self): #For the circle + cross fixation (see Thaler, Shutz, Goodale & Gegenfurtner (2013))
        shell()
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

    def example_tones(self): 
        self.firstscreen.draw()
        self.screen.flip()
        event.waitKeys('spacebar')

        if self.session.background == 'staircase':
            # Play separate tones and noise (only before run 0)
            if self.task == 'detect':
                self.target_tone_first.draw()
            elif self.task == 'discrim':
                self.low_tone_first.draw()
            self.draw_fixation()
            self.screen.flip()

            for i in range(0,3):
                self.target1 = sound.backend_sounddevice.SoundDeviceSound('target1.wav',stereo=True,volume=1.0)
                self.target1.setVolume(0.5)
                self.target1.play()
                core.wait(0.5)
                core.wait(1)

            core.wait(1)
        
            if self.task == 'discrim':
                self.high_tone_first.draw()
                self.draw_fixation()
                self.screen.flip()
                for i in range(0,3):
                    self.target2 = sound.backend_sounddevice.SoundDeviceSound('target2.wav', stereo=True,volume=1.0)
                    self.target2.setVolume(0.5)
                    self.target2.play()
                    core.wait(0.5)
                    core.wait(1)

            core.wait(1)

            self.noise_tone_first.draw()
            self.draw_fixation()
            self.screen.flip()
            for i in range(0,3):
                self.noise.play()
                core.wait(0.5)
                core.wait(1)

            core.wait(1)
            self.target_tone.setColor = ([-1,-1,-1])
            self.high_tone.setColor = ([-1,-1,-1])
            self.low_tone.setColor = ([-1,-1,-1])

        # Example tones so participants are reminded of the template.
        if self.task == 'detect':
            self.target_tone.draw()
        elif self.task == 'discrim':
            self.low_tone.draw()
        self.draw_fixation()
        self.screen.flip()

        for i in range(0,3):
            self.target1 = sound.backend_sounddevice.SoundDeviceSound('target1.wav', stereo=True,volume=1)
            self.target1.setVolume(0.2)
            self.noise.play()
            self.target1.play()
            core.wait(0.5)
            core.wait(1)

        core.wait(1)
        
        if self.task == 'discrim':
            self.high_tone.draw()
            self.draw_fixation()
            self.screen.flip()
            for i in range(0,3):
                self.target2 = sound.backend_sounddevice.SoundDeviceSound('target2.wav', stereo=True,volume=1)
                self.target2.setVolume(0.2)
                self.noise.play()
                self.target2.play()
                core.wait(0.5)
                core.wait(1)

        self.draw_fixation()
        self.screen.flip()

        self.message.draw()
        self.screen.flip()
    #Not so neat, but otherwise it gets stuck.
        for ev in event.waitKeys(keyList='spacebar'):
            if ev:
                self.phase_forward()

    def draw(self):
        if self.phase == 0: # 
            if self.session.background != 'staircase':
                self.line1.lineColor = self.session.background
                self.line2.lineColor = self.session.background
                self.fixation1.lineColor = self.session.background
            if self.ID % self.session.block_length == 0:
                if self.ID > 0:
                    self.feedback.draw()
                    self.screen.flip()
                    event.waitKeys('spacebar')
                self.example_tones()
            else:
                self.draw_fixation()

        if self.phase == 1: # Baseline
            if self.session.background != 'staircase':
                self.line1.lineColor = self.session.background
                self.line2.lineColor = self.session.background
                self.fixation1.lineColor = self.session.background
            self.draw_fixation()

        if self.phase ==2:  # Stimulus presentation
            self.target1 = sound.backend_sounddevice.SoundDeviceSound('target1.wav', stereo=True,volume=1)
            self.target2 = sound.backend_sounddevice.SoundDeviceSound('target2.wav', stereo=True,volume=1)
            self.target1.setVolume(float(self.session.kb.dv))
            self.target2.setVolume(float(self.session.kb.dv)) 

            print('Volume: ', str(self.session.kb.dv))
            self.parameters['signal_volume']=float(self.session.kb.dv)
            self.draw_fixation()
            if not self.noise_played:
                self.noise.play(loops=None)
                self.noise_played = True

            #to turn task into discrimination       
            if self.task == 'detect' and self.parameters['signal_present'] == 1:
                if not self.signal_played:
                    self.target1.play()
                    self.signal_played = True
            elif self.task == 'discrim':
                if self.parameters['disc_stim'] == 1:
                    if not self.signal_played:
                        self.target1.play()
                        self.signal_played = True  
                elif self.parameters['disc_stim'] == 2:
                    if not self.signal_played:
                        self.target2.play()
                        self.signal_played = True
                        
            if self.too_early: #turn red if subjects press during phase 2
                self.draw_red_fixation()
            
        elif self.phase == 3:# decision interval
            if self.too_early:
                self.draw_red_fixation()
            else:
                self.draw_fixation() 

        elif self.phase == 4: # ITI
            if self.parameters['correct'] == -1:
                self.draw_red_fixation()
                core.wait(.5)
                self.draw_fixation()

            self.draw_fixation()

        super(DetectTrial, self).draw()

    def event(self):
        trigger = None
        for ev in event.getKeys():
            if len(ev) > 0:
                if ev in ['esc', 'escape']:
                    self.events.append(
                        [-99, clock.getTime() - self.start_time])
                    self.stopped = True
                    self.session.stopped = True
                    print('run canceled by user')

                elif ev == 'space':
                    self.events.append(
                        [99, clock.getTime() - self.start_time])
                    if self.phase == 0:
                        self.phase_forward()

                if ev in ['a','s','k','l'] and self.phase == 2:
                    self.too_early = True
                    self.parameters['confidence'] = -2
                    self.parameters['answer'] = -2
                    self.parameters['correct'] = -2

                if ev in ['a','s','k','l'] and self.phase == 3: 

                    if self.too_early and self.session.background == 'staircase': #reset too_early, else the staircase won't proceed at all.
                        self.too_early = False 
                    if not self.too_early: 
                        self.events.append([1,clock.getTime()-self.start_time])
                        self.parameters['RT'] = clock.getTime() - self.stim_time
                        #confidence is always the same
                        if ev in ['a','l']:
                            self.parameters['confidence'] = 1
                        elif ev in ['s','k']:
                            self.parameters['confidence'] = 0

                        if self.task == 'discrim':
                            if self.version == 1:
                                if ev in ['a','s']:
                                    self.parameters['answer'] = 1
                                elif ev in ['k','l']:
                                    self.parameters['answer'] = 2
                            elif self.version == 2:
                                if ev in ['a','s']:
                                    self.parameters['answer'] = 2
                                elif ev in ['k','l']:
                                    self.parameters['answer'] = 1
                            if self.parameters['answer'] == self.parameters['disc_stim']:
                               self.parameters['correct'] = 1
                            else:
                               self.parameters['correct'] = 0
                            print('Correct: ', self.parameters['correct'])

                            # Update the volume according to Kaernbach. If not answered or answered too early - don't update the volume. 
                            if self.parameters['correct'] > -1:
                                self.session.kb.trial(self.parameters['correct'])

                            self.phase_forward()

                        elif self.task == 'detect': 
                            if self.version == 1:    
                                if ev in ['a','s']:
                                    self.parameters['answer'] = 0
                                elif ev in ['k','l']:
                                    self.parameters['answer'] = 1
                            elif self.version == 2:
                                if ev in ['a','s']:
                                    self.parameters['answer'] = 1
                                elif ev in ['k','l']:
                                    self.parameters['answer'] = 0
                            if self.parameters['answer'] == self.parameters['signal_present']:
                               self.parameters['correct'] = 1
                            else:
                               self.parameters['correct'] = 0
                            print('Correct: ', self.parameters['correct'])

                            # Update the volume according to Kaernbach. If not answered or answered too early - don't update the volume. 
                            if self.parameters['correct'] > -1:
                                self.session.kb.trial(self.parameters['correct'])

                            self.phase_forward()
                     
            super(DetectTrial, self).key_event(event)
        
    def run(self):

        self.start_time = self.session.clock.getTime()
        if tracker_on:
            self.tracker.log('trial ' + str(self.ID) + ' started at ' + str(self.start_time) )
            self.tracker.send_command('record_status_message "Trial ' + str(self.ID) + '"')
        self.events.append('trial ' + str(self.ID) + ' started at ' + str(self.start_time))

        trigger = None
        restart = 0 
        #for self.thisVolume in self.staircase:

        while not self.stopped:
        #for self.thisVolume in self.staircase: 
            self.run_time = clock.getTime() - self.start_time
                            
            if self.phase == 0:
                self.prestimulation_time = clock.getTime()
                # For all trials that are not FTIB, skip phase 0
                if self.ID != 0 and self.ID % self.session.block_length != 0:
                    if ( self.prestimulation_time  - self.start_time ) > self.phase_durations[0]:
                        self.phase_forward()
                
            elif self.phase == 1:  # pre-stim cue; phase is timed
                self.delay_1_time = clock.getTime()

                if ( self.delay_1_time - self.prestimulation_time ) > self.phase_durations[1]:
                    self.phase_forward()

            elif self.phase == 2:  # stimulus presentation; phase is timed
                self.stim_time = clock.getTime()
                    
                if ( self.stim_time - self.delay_1_time ) > self.phase_durations[2]: 
                    self.phase_forward()

            elif self.phase == 3:  # Decision interval; phase is timed, but aborted at response
                self.answer_time = clock.getTime()
                if self.parameters['answer'] != -1 and self.parameters['answer'] != -2: #end phase when respond
                    self.phase_forward()
                if self.session.background != 'staircase': #only end phase after some time when no response if this is not the staircase!   
                    if ( self.answer_time  - self.stim_time) > self.phase_durations[3]: #end phase after some time when no response
                        self.phase_forward()

            elif self.phase == 4: #ITI

                self.ITI_time = clock.getTime()
                self.too_early = False # Reset for next trial
                
                if ( self.ITI_time - self.answer_time ) > self.phase_durations[4]:
                    self.stopped = True
                    self.stop()
                    return 
            
            # events and draw:
            self.event()
            self.draw()
                        
        # we have stopped:
        self.stop()

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
                if not corr:
                    if self.trialn <= self.factor + 1:
                        self.reversals[0] += 2
                        self.initialerrfix = False

            if corr:
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
            self.prevcorr = corr
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

class DetectSession(EyelinkSession):
    def __init__(self, subject_initials, task, nr_trials, block_length, background, tracker_on=False, use_parallel=False, miniblock=0):
        super(DetectSession, self).__init__(subject_initials,background)
        config_file = os.path.join(os.path.abspath(os.getcwd()), 'default_settings.json')
        self.create_screen(  size=[1920, 1080],full_screen = fullscr, background_color = (0, 0, 0), physical_screen_distance = 80, engine = 'pygaze') #,  ,
        # screen = monitors.Monitor('testMonitor')
        # screen.setSizePix([1920,1080])
        # screen.setWidth(52.1)
        # screen.setDistance(57)

        # self.my_monitor = monitors.Monitor(name='mymon')
        # self.my_monitor.setSizePix((1920, 1080))
        # self.my_monitor.setWidth(20)

        self.block_length = block_length
        self.nr_trials = nr_trials
        self.background = background
        self.task = task 
        self.subject_initials = initials

        if tracker_on:
            pygaze.settings.EVENTDETECTION = 'native'
            self.create_tracker(sensitivity_class = 1, sample_rate=500) #
            print self.tracker.connected()
            self.tracker_setup(sensitivity_class = 1, sample_rate=500)

        #Took out self.create_output_filename() because I don't like the filenames with the date and time in it.
        datadir = 'data/' + self.task + '/' + str(self.subject_initials) + '/'
        
        if not os.path.exists(datadir):
            os.makedirs(datadir)     

        #if a file already exists, add a number to the file until it doesn't exist. 
        if not os.path.isfile(datadir + str(self.subject_initials) + '_' + str(self.task) + '_' + str(self.background) + '_outputDict.pkl'):
            self.output_file = os.path.join(datadir + str(self.subject_initials) + '_' + self.task + '_' + str(self.background))  
        else:
            i=1
            while True:
                if not os.path.isfile(datadir + str(self.subject_initials) + '_' + self.task + '_' + str(self.background) + '_' + str(i) + '_outputDict.pkl'):
                    self.output_file = os.path.join(datadir + str(self.subject_initials) + '_' + self.task + '_' + str(self.background) + '_' + str(i))
                    break
                i += 1  

        self.miniblock=miniblock

        self.use_parallel = use_parallel
        self.p_width = 5/float(1000)
        # Trigger values
        run_start_trigger = {'discrim':125 , 'detect':126}
        
        self.p_run_start = run_start_trigger[self.task]
        self.p_run_end = 126                        #
        self.p_cue = 5                              #
        self.p_stimulus_left = 8                    #
        self.p_stimulus_right = 9                   #
        self.p_stimulus_noise = 10                  #
        self.p_choice_left_sure = 16                #
        self.p_choice_left_unsure = 17              #
        self.p_choice_right_sure = 18               #
        self.p_choice_right_unsure = 19             #
        self.p_feedback_sound = 32                  #
        self.p_feedback_visual = 33 

        # create staircase object
        if self.background == 'staircase': 
            dv0 = .3
            reversals = [2,400]
            stepsizes = (.05,.005)
        else:
            try: 
                dv0 = np.array([np.loadtxt(os.getcwd() + '/data/' + self.task + '_staircase/participant_' + initials + '/' + initials + '_' + self.task + '_threshold.txt')]) 
            except:
                raise NameError('no staircase data for participant')  
            print(dv0)
            if dv0 < 0.01:
                print('Staircase value is zero, raising to .03')
                dv0 = 0.03
            reversals = [nr_trials+1,nr_trials+1]
            stepsizes = [.005,.005]

        self.kb = Kaernbach1991(subject_initials=self.subject_initials, task=self.task, dv0=dv0, p=0.75, reversals=reversals, stepsizes=stepsizes, initialerrfix=True, avgrevsonly=True, cap=False)

        self.create_yes_no_trials()

    def create_yes_no_trials(self):
        self.signal_present = np.array([0,1])
        self.disc_stim = np.array([1,2])
        # standard parameters (same for all trials), signal volume will be updated every trial:
        self.standard_parameters = {'signal_volume': None,
                                    'miniblock': self.miniblock,
                                    }
        
        # create yes-no trials in nested for-loop:
        self.trial_parameters_and_durs = []    
        self.trial_counter = 0
        self.total_duration = 0

        for i in range(self.nr_trials/self.signal_present.shape[0]):
            for j in range(self.signal_present.shape[0]):

                # phase durations, and iti's:
                phase_durs = [-0.01, 0.6, 0.5, 1.5, np.random.uniform(0,0.4)]
                params = self.standard_parameters
                params.update({'signal_present': self.signal_present[j]})
                params.update({'disc_stim': self.disc_stim[j]})

                self.trial_parameters_and_durs.append([params.copy(), np.array(phase_durs)])
                self.total_duration += np.array(phase_durs).sum()
                self.trial_counter += 1

        self.run_order = np.argsort(np.random.rand(len(self.trial_parameters_and_durs)))

        # print params:
        print("number trials: %i." % self.trial_counter)
        # if self.trial_counter != NR_TRIALS:
        #     raise ValueError('number of created trials does not match pre-defined number of trials')

        print("total duration: %.2f min." % (self.total_duration / 60.0))

    def run(self):
        """run the session"""
        # cycle through trials
        self.corrects = []
        self.confidence = []
        self.clock = clock
        
        if tracker_on:
            self.tracker.status_msg('run started at ' + str(clock.getTime()) + ' trigger ' + str(self.p_run_start) )
        
        self.start_time = clock.getTime()
                
        # Display black - white - black screens to determine pupil size limits
        self.center = (self.screen.size[0]/2.0, self.screen.size[1]/2.0)
        self.fixation = GratingStim(self.screen, mask = 'circle',size=4, pos=[0,0], sf=0, color ='grey')
        self.baseline_instruct = TextStim(self.screen, text = 'please keep your focus on the dot in the middle', pos = (0,50), color = (0,0,0), height=20)

        if self.miniblock==0:
            
            t0 = clock.getTime()
            if tracker_on:
                while clock.getTime() - t0 < 3:
                    self.baseline_instruct.draw()
                    self.screen.flip()
            
            t1 = clock.getTime()
            if tracker_on:
                self.tracker.status_msg('pupil baseline 1 started ' + str(clock.getTime())  )
                while clock.getTime() - t1 < 15:
                    self.screen.color=(-1,-1,-1)
                    self.fixation.draw()
                    self.screen.flip()
            if tracker_on:
                self.tracker.status_msg('pupil baseline 1 ended ' + str(clock.getTime())  )     

            t2 = clock.getTime()
            if tracker_on:
                self.tracker.status_msg('pupil baseline 2 started ' + str(clock.getTime())  )               
                while clock.getTime() - t2 < 15:
                    self.screen.color=(1,1,1)
                    self.fixation.draw()
                    self.screen.flip()  
            if tracker_on:
                self.tracker.status_msg('pupil baseline 2 ended ' + str(clock.getTime())  )
                
            t3 = clock.getTime()
            if tracker_on:
                self.tracker.status_msg('pupil baseline 3 started ' + str(clock.getTime())  )
                while clock.getTime() - t3 < 15:
                    self.screen.color=(-1,-1,-1)
                    self.fixation.draw()
                    self.screen.flip()
                if tracker_on:
                    self.tracker.status_msg('pupil baseline 3 ended ' + str(clock.getTime())  )
                
            self.screen.color=(0.5, 0.5, 0.5)
            self.screen.flip()
        
        if self.background == 'staircase':  #XXX dit moet ergens anders
            self.screen.color = (0,0,0)
        else:
            self.screen.color = self.background

        self.trial_counter = 0

        while self.trial_counter < self.nr_trials:
            print('Trial: ', self.trial_counter)
            if self.trial_counter == self.nr_trials: 
                this_trial = DetectTrial(task=self.task, parameters=self.trial_parameters_and_durs[self.run_order[self.trial_counter]][0], phase_durations=[-0.01, 0.6, 0.5, 2.5, 4 ], session=self, screen=self.screen, tracker=self.tracker, ID=self.trial_counter)
            else:    
                this_trial = DetectTrial(task=self.task, parameters=self.trial_parameters_and_durs[self.run_order[self.trial_counter]][0], phase_durations=self.trial_parameters_and_durs[self.run_order[self.trial_counter]][1], session=self, screen=self.screen, tracker=self.tracker, ID=self.trial_counter)
            this_trial.run()    
            self.corrects.append(this_trial.parameters['correct'])
            self.confidence.append(this_trial.parameters['confidence']) 

            if self.stopped == True:
                break
            self.trial_counter += 1
        
        self.stop_time = clock.getTime()

        pygame.mixer.quit()
        #self.screen.close()
        parsopf = open(self.output_file + '_outputDict.pkl', 'a')
        pkl.dump(self.outputDict,parsopf)
        parsopf.close()
        # also output parameters as tsv
        opd = pd.DataFrame.from_records(self.outputDict['parameterArray'])
        opd.to_csv(path_or_buf=self.output_file + '.tsv', sep='\t', encoding='utf-8')

        if self.background != 'staircase':
            self.goodbye = visual.TextStim(self.screen, pos=[0,0], text='This is the end of this block.\nThe screen will now turn grey.',color=(0,0,0),wrapWidth=50,font='avenir')
            early = len(np.array(self.corrects)[-self.block_length:][np.array(self.corrects)[-self.block_length:] == -2]) / float(self.block_length) * 100.0
            misses = len(np.array(self.corrects)[-self.block_length:][np.array(self.corrects)[-self.block_length:] == -1]) / float(self.block_length) * 100.0
            conf = np.array(self.confidence)[-self.block_length:][np.array(self.confidence)[-self.block_length:] >= 0].sum() / float(self.block_length) * 100.0

            feedback_text = """You were more confident in %i%% of the trials.\n\nYou responded too quickly to %i%% of trials and you missed %i%% of trials. Please try to keep this to a minimum.\n\nPress the spacebar to continue.""" % (conf,early,misses)    
            feedback_final = """You were more confident in %i%% of the trials.\n\nYou responded too quickly to %i%% of trials and you missed %i%% of trials. Please try to keep this to a minimum.\n\nThis screen will close automatically.""" % (conf,early,misses)
            print(str(conf) + ' percent high conf')
            print(str(early) + ' percent fast response trials')
            print(str(misses) + ' percent missed trials')

            if self.stopped: 
                self.final_feedback = visual.TextStim(self.screen, font='avenir', pos=[0,0],text=feedback_final, color=(0,0,0),wrapWidth=40)
                self.final_feedback.draw()
                self.screen.flip()
                core.wait(8)
                self.screen.close()
            else:
                self.final_feedback = visual.TextStim(self.screen, font='avenir', pos=[0,0],text=feedback_text, color=(0,0,0),wrapWidth=40) 
                self.final_feedback.draw()
                self.screen.flip()
                event.waitKeys('spacebar')
        else: 
            self.goodbye = visual.TextStim(self.screen, pos=[0,0], text='This is the end of the staircase procedure.',color=(-1,-1,-1),wrapWidth=50,font='avenir')

        self.goodbye.draw()
        self.screen.flip()
        core.wait(2)
        if self.tracker_on:
            self.tracker.status_msg('run ended at ' + str(clock.getTime()) + ' trigger ' + str(self.p_run_end) )
        print('elapsed time: %.2fs' %(self.stop_time-self.start_time))      

        self.breakscreen = visual.TextStim(self.screen, pos=[0,0], text='You can now take a break and take your head of the chinrest.\n\nPress the spacebar when you are finished taking a break.',font='avenir',color=(-1,-1,-1),wrapWidth=40)

        self.screen.color = 'grey'
        self.screen.flip()
        self.breakscreen.draw()
        self.screen.flip()
        event.waitKeys('spacebar')
        self.screen.clearBuffer
 
def main(initials,block_length,nr_trials):

    prestaircase = DetectSession(subject_initials=initials, nr_trials=nr_staircase_trials, block_length =40,  background='staircase', tracker_on=False, use_parallel=False, task='detect', miniblock=1)
    prestaircase.run()

    condition = [['black','detect'],
                 ['black','discrim'],
                 ['white','detect'],
                 ['white','discrim']]
    np.random.shuffle(condition)

    for i in range(4):
        task = condition[i][1]
        print(task)
        background=condition[i][0]
        print(background)
        miniblock=i
        print(miniblock)

        ts = DetectSession(subject_initials=initials, nr_trials=NR_TRIALS, block_length = block_length,  background=background, tracker_on=tracker_on, use_parallel=use_parallel, task=task, miniblock=miniblock)
        ts.run()

        if ts.stopped:
            ts.close()
            ts.screen.close()
            break

        if not os.path.exists('data/' + task + '/' + initials + '/'):
            os.makedirs('data/' + task +'/' + initials + '/')

if __name__ == '__main__':
    # Store info about the experiment session
    initials = raw_input('Participant: ')
    main(initials=initials, block_length = block_length, nr_trials=NR_TRIALS)
