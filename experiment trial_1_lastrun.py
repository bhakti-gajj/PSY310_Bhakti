#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.1.1),
    on Sun Nov 23 14:07:53 2025
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.1.1'
expName = 'experiment trial_1'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/bhaktigajjar/Desktop/AhdUni/lap psychology/attentionalBlink_SoloProject/experiment trial_1_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('instr_key') is None:
        # initialise instr_key
        instr_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_key',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('end_key') is None:
        # initialise end_key
        end_key = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_key',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions" ---
    instrText = visual.TextStim(win=win, name='instrText',
        text='You will see a fast stream of letters. Two numbers in the stream are the targets (T1 and T2). \nAfter the stream, type the FIRST target (T1) and press Enter, then type the SECOND target (T2) and press Enter.\nPress SPACE to begin.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_key = keyboard.Keyboard(deviceName='instr_key')
    
    # --- Initialize components for Routine "trial" ---
    fixation = visual.ShapeStim(
        win=win, name='fixation', vertices='cross',
        size=(0.1, 0.1),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    item01 = visual.TextStim(win=win, name='item01',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    item02 = visual.TextStim(win=win, name='item02',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    item03 = visual.TextStim(win=win, name='item03',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    item04 = visual.TextStim(win=win, name='item04',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    item05 = visual.TextStim(win=win, name='item05',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    item06 = visual.TextStim(win=win, name='item06',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-6.0);
    item07 = visual.TextStim(win=win, name='item07',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-7.0);
    item08 = visual.TextStim(win=win, name='item08',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-8.0);
    item09 = visual.TextStim(win=win, name='item09',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-9.0);
    item10 = visual.TextStim(win=win, name='item10',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-10.0);
    item11 = visual.TextStim(win=win, name='item11',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-11.0);
    item12 = visual.TextStim(win=win, name='item12',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-12.0);
    item13 = visual.TextStim(win=win, name='item13',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-13.0);
    item14 = visual.TextStim(win=win, name='item14',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-14.0);
    item15 = visual.TextStim(win=win, name='item15',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-15.0);
    item16 = visual.TextStim(win=win, name='item16',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-16.0);
    
    # --- Initialize components for Routine "T1_response" ---
    respInstr = visual.TextStim(win=win, name='respInstr',
        text='Type the FIRST target (T1) and press Enter. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    t1Prompt = visual.TextStim(win=win, name='t1Prompt',
        text='T1:\n',
        font='Arial',
        pos=(-0.3, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    textbox = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Arial',
         ori=0.0, pos=(-0.1, -0.1), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox',
         depth=-2, autoLog=True,
    )
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "T2_response" ---
    resplnstr_2 = visual.TextStim(win=win, name='resplnstr_2',
        text='Type the SECOND target (T2) and press Enter. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    t2Prompt = visual.TextStim(win=win, name='t2Prompt',
        text='T2:\n',
        font='Arial',
        pos=(-0.3, -0.1), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    textbox_2 = visual.TextBox2(
         win, text=None, placeholder='Type here...', font='Arial',
         ori=0.0, pos=(-0.1, -0.1), draggable=False,      letterHeight=0.05,
         size=(0.5, 0.5), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, languageStyle='LTR',
         editable=True,
         name='textbox_2',
         depth=-2, autoLog=True,
    )
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "end" ---
    thanks = visual.TextStim(win=win, name='thanks',
        text='Thanks! Press any key to finish.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_key = keyboard.Keyboard(deviceName='end_key')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[instrText, instr_key],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_key
    instr_key.keys = []
    instr_key.rt = []
    _instr_key_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instrText* updates
        
        # if instrText is starting this frame...
        if instrText.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instrText.frameNStart = frameN  # exact frame index
            instrText.tStart = t  # local t and not account for scr refresh
            instrText.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instrText, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instrText.started')
            # update status
            instrText.status = STARTED
            instrText.setAutoDraw(True)
        
        # if instrText is active this frame...
        if instrText.status == STARTED:
            # update params
            pass
        
        # *instr_key* updates
        waitOnFlip = False
        
        # if instr_key is starting this frame...
        if instr_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_key.frameNStart = frameN  # exact frame index
            instr_key.tStart = t  # local t and not account for scr refresh
            instr_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_key.started')
            # update status
            instr_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_key.status == STARTED and not waitOnFlip:
            theseKeys = instr_key.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_key_allKeys.extend(theseKeys)
            if len(_instr_key_allKeys):
                instr_key.keys = _instr_key_allKeys[-1].name  # just the last key pressed
                instr_key.rt = _instr_key_allKeys[-1].rt
                instr_key.duration = _instr_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if instr_key.keys in ['', [], None]:  # No response was made
        instr_key.keys = None
    thisExp.addData('instr_key.keys',instr_key.keys)
    if instr_key.keys != None:  # we had a response
        thisExp.addData('instr_key.rt', instr_key.rt)
        thisExp.addData('instr_key.duration', instr_key.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trialsLoop = data.TrialHandler2(
        name='trialsLoop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('attentional_blink_12trials.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trialsLoop)  # add the loop to the experiment
    thisTrialsLoop = trialsLoop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrialsLoop.rgb)
    if thisTrialsLoop != None:
        for paramName in thisTrialsLoop:
            globals()[paramName] = thisTrialsLoop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrialsLoop in trialsLoop:
        trialsLoop.status = STARTED
        if hasattr(thisTrialsLoop, 'status'):
            thisTrialsLoop.status = STARTED
        currentLoop = trialsLoop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrialsLoop.rgb)
        if thisTrialsLoop != None:
            for paramName in thisTrialsLoop:
                globals()[paramName] = thisTrialsLoop[paramName]
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[fixation, item01, item02, item03, item04, item05, item06, item07, item08, item09, item10, item11, item12, item13, item14, item15, item16],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        item01.setText(stim01
        
        )
        item02.setText(stim02
        
        )
        item03.setText(stim03
        
        )
        item04.setText(stim04
        
        )
        item05.setText(stim05
        
        )
        item06.setText(stim06
        
        )
        item07.setText(stim07
        
        )
        item08.setText(stim08
        
        )
        item09.setText(stim09
        
        )
        item10.setText(stim10
        
        )
        item11.setText(stim11
        
        )
        item12.setText(stim12
        
        )
        item13.setText(stim13
        
        )
        item14.setText(stim14
        
        )
        item15.setText(stim15
        
        )
        item16.setText(stim16
        
        )
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial" ---
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 4.5:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsLoop, 'status') and thisTrialsLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            
            # *item01* updates
            
            # if item01 is starting this frame...
            if item01.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                item01.frameNStart = frameN  # exact frame index
                item01.tStart = t  # local t and not account for scr refresh
                item01.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item01, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item01.started')
                # update status
                item01.status = STARTED
                item01.setAutoDraw(True)
            
            # if item01 is active this frame...
            if item01.status == STARTED:
                # update params
                pass
            
            # if item01 is stopping this frame...
            if item01.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item01.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item01.tStop = t  # not accounting for scr refresh
                    item01.tStopRefresh = tThisFlipGlobal  # on global time
                    item01.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item01.stopped')
                    # update status
                    item01.status = FINISHED
                    item01.setAutoDraw(False)
            
            # *item02* updates
            
            # if item02 is starting this frame...
            if item02.status == NOT_STARTED and tThisFlip >= 0.75-frameTolerance:
                # keep track of start time/frame for later
                item02.frameNStart = frameN  # exact frame index
                item02.tStart = t  # local t and not account for scr refresh
                item02.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item02, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item02.started')
                # update status
                item02.status = STARTED
                item02.setAutoDraw(True)
            
            # if item02 is active this frame...
            if item02.status == STARTED:
                # update params
                pass
            
            # if item02 is stopping this frame...
            if item02.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item02.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item02.tStop = t  # not accounting for scr refresh
                    item02.tStopRefresh = tThisFlipGlobal  # on global time
                    item02.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item02.stopped')
                    # update status
                    item02.status = FINISHED
                    item02.setAutoDraw(False)
            
            # *item03* updates
            
            # if item03 is starting this frame...
            if item03.status == NOT_STARTED and tThisFlip >= 1.0-frameTolerance:
                # keep track of start time/frame for later
                item03.frameNStart = frameN  # exact frame index
                item03.tStart = t  # local t and not account for scr refresh
                item03.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item03, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item03.started')
                # update status
                item03.status = STARTED
                item03.setAutoDraw(True)
            
            # if item03 is active this frame...
            if item03.status == STARTED:
                # update params
                pass
            
            # if item03 is stopping this frame...
            if item03.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item03.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item03.tStop = t  # not accounting for scr refresh
                    item03.tStopRefresh = tThisFlipGlobal  # on global time
                    item03.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item03.stopped')
                    # update status
                    item03.status = FINISHED
                    item03.setAutoDraw(False)
            
            # *item04* updates
            
            # if item04 is starting this frame...
            if item04.status == NOT_STARTED and tThisFlip >= 1.25-frameTolerance:
                # keep track of start time/frame for later
                item04.frameNStart = frameN  # exact frame index
                item04.tStart = t  # local t and not account for scr refresh
                item04.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item04, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item04.started')
                # update status
                item04.status = STARTED
                item04.setAutoDraw(True)
            
            # if item04 is active this frame...
            if item04.status == STARTED:
                # update params
                pass
            
            # if item04 is stopping this frame...
            if item04.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item04.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item04.tStop = t  # not accounting for scr refresh
                    item04.tStopRefresh = tThisFlipGlobal  # on global time
                    item04.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item04.stopped')
                    # update status
                    item04.status = FINISHED
                    item04.setAutoDraw(False)
            
            # *item05* updates
            
            # if item05 is starting this frame...
            if item05.status == NOT_STARTED and tThisFlip >= 1.50-frameTolerance:
                # keep track of start time/frame for later
                item05.frameNStart = frameN  # exact frame index
                item05.tStart = t  # local t and not account for scr refresh
                item05.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item05, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item05.started')
                # update status
                item05.status = STARTED
                item05.setAutoDraw(True)
            
            # if item05 is active this frame...
            if item05.status == STARTED:
                # update params
                pass
            
            # if item05 is stopping this frame...
            if item05.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item05.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item05.tStop = t  # not accounting for scr refresh
                    item05.tStopRefresh = tThisFlipGlobal  # on global time
                    item05.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item05.stopped')
                    # update status
                    item05.status = FINISHED
                    item05.setAutoDraw(False)
            
            # *item06* updates
            
            # if item06 is starting this frame...
            if item06.status == NOT_STARTED and tThisFlip >= 1.75-frameTolerance:
                # keep track of start time/frame for later
                item06.frameNStart = frameN  # exact frame index
                item06.tStart = t  # local t and not account for scr refresh
                item06.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item06, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item06.started')
                # update status
                item06.status = STARTED
                item06.setAutoDraw(True)
            
            # if item06 is active this frame...
            if item06.status == STARTED:
                # update params
                pass
            
            # if item06 is stopping this frame...
            if item06.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item06.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item06.tStop = t  # not accounting for scr refresh
                    item06.tStopRefresh = tThisFlipGlobal  # on global time
                    item06.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item06.stopped')
                    # update status
                    item06.status = FINISHED
                    item06.setAutoDraw(False)
            
            # *item07* updates
            
            # if item07 is starting this frame...
            if item07.status == NOT_STARTED and tThisFlip >= 2.00-frameTolerance:
                # keep track of start time/frame for later
                item07.frameNStart = frameN  # exact frame index
                item07.tStart = t  # local t and not account for scr refresh
                item07.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item07, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item07.started')
                # update status
                item07.status = STARTED
                item07.setAutoDraw(True)
            
            # if item07 is active this frame...
            if item07.status == STARTED:
                # update params
                pass
            
            # if item07 is stopping this frame...
            if item07.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item07.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item07.tStop = t  # not accounting for scr refresh
                    item07.tStopRefresh = tThisFlipGlobal  # on global time
                    item07.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item07.stopped')
                    # update status
                    item07.status = FINISHED
                    item07.setAutoDraw(False)
            
            # *item08* updates
            
            # if item08 is starting this frame...
            if item08.status == NOT_STARTED and tThisFlip >= 2.25-frameTolerance:
                # keep track of start time/frame for later
                item08.frameNStart = frameN  # exact frame index
                item08.tStart = t  # local t and not account for scr refresh
                item08.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item08, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item08.started')
                # update status
                item08.status = STARTED
                item08.setAutoDraw(True)
            
            # if item08 is active this frame...
            if item08.status == STARTED:
                # update params
                pass
            
            # if item08 is stopping this frame...
            if item08.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item08.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item08.tStop = t  # not accounting for scr refresh
                    item08.tStopRefresh = tThisFlipGlobal  # on global time
                    item08.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item08.stopped')
                    # update status
                    item08.status = FINISHED
                    item08.setAutoDraw(False)
            
            # *item09* updates
            
            # if item09 is starting this frame...
            if item09.status == NOT_STARTED and tThisFlip >= 2.50-frameTolerance:
                # keep track of start time/frame for later
                item09.frameNStart = frameN  # exact frame index
                item09.tStart = t  # local t and not account for scr refresh
                item09.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item09, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item09.started')
                # update status
                item09.status = STARTED
                item09.setAutoDraw(True)
            
            # if item09 is active this frame...
            if item09.status == STARTED:
                # update params
                pass
            
            # if item09 is stopping this frame...
            if item09.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item09.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item09.tStop = t  # not accounting for scr refresh
                    item09.tStopRefresh = tThisFlipGlobal  # on global time
                    item09.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item09.stopped')
                    # update status
                    item09.status = FINISHED
                    item09.setAutoDraw(False)
            
            # *item10* updates
            
            # if item10 is starting this frame...
            if item10.status == NOT_STARTED and tThisFlip >= 2.75-frameTolerance:
                # keep track of start time/frame for later
                item10.frameNStart = frameN  # exact frame index
                item10.tStart = t  # local t and not account for scr refresh
                item10.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item10, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item10.started')
                # update status
                item10.status = STARTED
                item10.setAutoDraw(True)
            
            # if item10 is active this frame...
            if item10.status == STARTED:
                # update params
                pass
            
            # if item10 is stopping this frame...
            if item10.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item10.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item10.tStop = t  # not accounting for scr refresh
                    item10.tStopRefresh = tThisFlipGlobal  # on global time
                    item10.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item10.stopped')
                    # update status
                    item10.status = FINISHED
                    item10.setAutoDraw(False)
            
            # *item11* updates
            
            # if item11 is starting this frame...
            if item11.status == NOT_STARTED and tThisFlip >= 3.00-frameTolerance:
                # keep track of start time/frame for later
                item11.frameNStart = frameN  # exact frame index
                item11.tStart = t  # local t and not account for scr refresh
                item11.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item11, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item11.started')
                # update status
                item11.status = STARTED
                item11.setAutoDraw(True)
            
            # if item11 is active this frame...
            if item11.status == STARTED:
                # update params
                pass
            
            # if item11 is stopping this frame...
            if item11.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item11.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item11.tStop = t  # not accounting for scr refresh
                    item11.tStopRefresh = tThisFlipGlobal  # on global time
                    item11.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item11.stopped')
                    # update status
                    item11.status = FINISHED
                    item11.setAutoDraw(False)
            
            # *item12* updates
            
            # if item12 is starting this frame...
            if item12.status == NOT_STARTED and tThisFlip >= 3.25-frameTolerance:
                # keep track of start time/frame for later
                item12.frameNStart = frameN  # exact frame index
                item12.tStart = t  # local t and not account for scr refresh
                item12.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item12, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item12.started')
                # update status
                item12.status = STARTED
                item12.setAutoDraw(True)
            
            # if item12 is active this frame...
            if item12.status == STARTED:
                # update params
                pass
            
            # if item12 is stopping this frame...
            if item12.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item12.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item12.tStop = t  # not accounting for scr refresh
                    item12.tStopRefresh = tThisFlipGlobal  # on global time
                    item12.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item12.stopped')
                    # update status
                    item12.status = FINISHED
                    item12.setAutoDraw(False)
            
            # *item13* updates
            
            # if item13 is starting this frame...
            if item13.status == NOT_STARTED and tThisFlip >= 3.50-frameTolerance:
                # keep track of start time/frame for later
                item13.frameNStart = frameN  # exact frame index
                item13.tStart = t  # local t and not account for scr refresh
                item13.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item13, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item13.started')
                # update status
                item13.status = STARTED
                item13.setAutoDraw(True)
            
            # if item13 is active this frame...
            if item13.status == STARTED:
                # update params
                pass
            
            # if item13 is stopping this frame...
            if item13.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item13.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item13.tStop = t  # not accounting for scr refresh
                    item13.tStopRefresh = tThisFlipGlobal  # on global time
                    item13.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item13.stopped')
                    # update status
                    item13.status = FINISHED
                    item13.setAutoDraw(False)
            
            # *item14* updates
            
            # if item14 is starting this frame...
            if item14.status == NOT_STARTED and tThisFlip >= 3.75-frameTolerance:
                # keep track of start time/frame for later
                item14.frameNStart = frameN  # exact frame index
                item14.tStart = t  # local t and not account for scr refresh
                item14.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item14, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item14.started')
                # update status
                item14.status = STARTED
                item14.setAutoDraw(True)
            
            # if item14 is active this frame...
            if item14.status == STARTED:
                # update params
                pass
            
            # if item14 is stopping this frame...
            if item14.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item14.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item14.tStop = t  # not accounting for scr refresh
                    item14.tStopRefresh = tThisFlipGlobal  # on global time
                    item14.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item14.stopped')
                    # update status
                    item14.status = FINISHED
                    item14.setAutoDraw(False)
            
            # *item15* updates
            
            # if item15 is starting this frame...
            if item15.status == NOT_STARTED and tThisFlip >= 4.00-frameTolerance:
                # keep track of start time/frame for later
                item15.frameNStart = frameN  # exact frame index
                item15.tStart = t  # local t and not account for scr refresh
                item15.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item15, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item15.started')
                # update status
                item15.status = STARTED
                item15.setAutoDraw(True)
            
            # if item15 is active this frame...
            if item15.status == STARTED:
                # update params
                pass
            
            # if item15 is stopping this frame...
            if item15.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item15.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item15.tStop = t  # not accounting for scr refresh
                    item15.tStopRefresh = tThisFlipGlobal  # on global time
                    item15.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item15.stopped')
                    # update status
                    item15.status = FINISHED
                    item15.setAutoDraw(False)
            
            # *item16* updates
            
            # if item16 is starting this frame...
            if item16.status == NOT_STARTED and tThisFlip >= 4.25-frameTolerance:
                # keep track of start time/frame for later
                item16.frameNStart = frameN  # exact frame index
                item16.tStart = t  # local t and not account for scr refresh
                item16.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(item16, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'item16.started')
                # update status
                item16.status = STARTED
                item16.setAutoDraw(True)
            
            # if item16 is active this frame...
            if item16.status == STARTED:
                # update params
                pass
            
            # if item16 is stopping this frame...
            if item16.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > item16.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    item16.tStop = t  # not accounting for scr refresh
                    item16.tStopRefresh = tThisFlipGlobal  # on global time
                    item16.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'item16.stopped')
                    # update status
                    item16.status = FINISHED
                    item16.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=trial,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial.maxDurationReached:
            routineTimer.addTime(-trial.maxDuration)
        elif trial.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-4.500000)
        
        # --- Prepare to start Routine "T1_response" ---
        # create an object to store info about Routine T1_response
        T1_response = data.Routine(
            name='T1_response',
            components=[respInstr, t1Prompt, textbox, key_resp],
        )
        T1_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textbox.reset()
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for T1_response
        T1_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        T1_response.tStart = globalClock.getTime(format='float')
        T1_response.status = STARTED
        thisExp.addData('T1_response.started', T1_response.tStart)
        T1_response.maxDuration = None
        # keep track of which components have finished
        T1_responseComponents = T1_response.components
        for thisComponent in T1_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "T1_response" ---
        T1_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsLoop, 'status') and thisTrialsLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *respInstr* updates
            
            # if respInstr is starting this frame...
            if respInstr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                respInstr.frameNStart = frameN  # exact frame index
                respInstr.tStart = t  # local t and not account for scr refresh
                respInstr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(respInstr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'respInstr.started')
                # update status
                respInstr.status = STARTED
                respInstr.setAutoDraw(True)
            
            # if respInstr is active this frame...
            if respInstr.status == STARTED:
                # update params
                pass
            
            # *t1Prompt* updates
            
            # if t1Prompt is starting this frame...
            if t1Prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t1Prompt.frameNStart = frameN  # exact frame index
                t1Prompt.tStart = t  # local t and not account for scr refresh
                t1Prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t1Prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't1Prompt.started')
                # update status
                t1Prompt.status = STARTED
                t1Prompt.setAutoDraw(True)
            
            # if t1Prompt is active this frame...
            if t1Prompt.status == STARTED:
                # update params
                pass
            
            # *textbox* updates
            
            # if textbox is starting this frame...
            if textbox.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox.frameNStart = frameN  # exact frame index
                textbox.tStart = t  # local t and not account for scr refresh
                textbox.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox.started')
                # update status
                textbox.status = STARTED
                textbox.setAutoDraw(True)
            
            # if textbox is active this frame...
            if textbox.status == STARTED:
                # update params
                pass
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=T1_response,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                T1_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in T1_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "T1_response" ---
        for thisComponent in T1_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for T1_response
        T1_response.tStop = globalClock.getTime(format='float')
        T1_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('T1_response.stopped', T1_response.tStop)
        trialsLoop.addData('textbox.text',textbox.text)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        trialsLoop.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            trialsLoop.addData('key_resp.rt', key_resp.rt)
            trialsLoop.addData('key_resp.duration', key_resp.duration)
        # the Routine "T1_response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "T2_response" ---
        # create an object to store info about Routine T2_response
        T2_response = data.Routine(
            name='T2_response',
            components=[resplnstr_2, t2Prompt, textbox_2, key_resp_2],
        )
        T2_response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        textbox_2.reset()
        # create starting attributes for key_resp_2
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # store start times for T2_response
        T2_response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        T2_response.tStart = globalClock.getTime(format='float')
        T2_response.status = STARTED
        thisExp.addData('T2_response.started', T2_response.tStart)
        T2_response.maxDuration = None
        # keep track of which components have finished
        T2_responseComponents = T2_response.components
        for thisComponent in T2_response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "T2_response" ---
        T2_response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTrialsLoop, 'status') and thisTrialsLoop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *resplnstr_2* updates
            
            # if resplnstr_2 is starting this frame...
            if resplnstr_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                resplnstr_2.frameNStart = frameN  # exact frame index
                resplnstr_2.tStart = t  # local t and not account for scr refresh
                resplnstr_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(resplnstr_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'resplnstr_2.started')
                # update status
                resplnstr_2.status = STARTED
                resplnstr_2.setAutoDraw(True)
            
            # if resplnstr_2 is active this frame...
            if resplnstr_2.status == STARTED:
                # update params
                pass
            
            # *t2Prompt* updates
            
            # if t2Prompt is starting this frame...
            if t2Prompt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                t2Prompt.frameNStart = frameN  # exact frame index
                t2Prompt.tStart = t  # local t and not account for scr refresh
                t2Prompt.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(t2Prompt, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 't2Prompt.started')
                # update status
                t2Prompt.status = STARTED
                t2Prompt.setAutoDraw(True)
            
            # if t2Prompt is active this frame...
            if t2Prompt.status == STARTED:
                # update params
                pass
            
            # *textbox_2* updates
            
            # if textbox_2 is starting this frame...
            if textbox_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                textbox_2.frameNStart = frameN  # exact frame index
                textbox_2.tStart = t  # local t and not account for scr refresh
                textbox_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(textbox_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'textbox_2.started')
                # update status
                textbox_2.status = STARTED
                textbox_2.setAutoDraw(True)
            
            # if textbox_2 is active this frame...
            if textbox_2.status == STARTED:
                # update params
                pass
            
            # *key_resp_2* updates
            waitOnFlip = False
            
            # if key_resp_2 is starting this frame...
            if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_2.frameNStart = frameN  # exact frame index
                key_resp_2.tStart = t  # local t and not account for scr refresh
                key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_2.started')
                # update status
                key_resp_2.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_2.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_2.getKeys(keyList=['return'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_2_allKeys.extend(theseKeys)
                if len(_key_resp_2_allKeys):
                    key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                    key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                    key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=T2_response,
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                T2_response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in T2_response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "T2_response" ---
        for thisComponent in T2_response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for T2_response
        T2_response.tStop = globalClock.getTime(format='float')
        T2_response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('T2_response.stopped', T2_response.tStop)
        trialsLoop.addData('textbox_2.text',textbox_2.text)
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        trialsLoop.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            trialsLoop.addData('key_resp_2.rt', key_resp_2.rt)
            trialsLoop.addData('key_resp_2.duration', key_resp_2.duration)
        # the Routine "T2_response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTrialsLoop as finished
        if hasattr(thisTrialsLoop, 'status'):
            thisTrialsLoop.status = FINISHED
        # if awaiting a pause, pause now
        if trialsLoop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            trialsLoop.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trialsLoop'
    trialsLoop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[thanks, end_key],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_key
    end_key.keys = []
    end_key.rt = []
    _end_key_allKeys = []
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *thanks* updates
        
        # if thanks is starting this frame...
        if thanks.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            thanks.frameNStart = frameN  # exact frame index
            thanks.tStart = t  # local t and not account for scr refresh
            thanks.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(thanks, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'thanks.started')
            # update status
            thanks.status = STARTED
            thanks.setAutoDraw(True)
        
        # if thanks is active this frame...
        if thanks.status == STARTED:
            # update params
            pass
        
        # *end_key* updates
        waitOnFlip = False
        
        # if end_key is starting this frame...
        if end_key.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_key.frameNStart = frameN  # exact frame index
            end_key.tStart = t  # local t and not account for scr refresh
            end_key.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_key, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_key.started')
            # update status
            end_key.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_key.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_key.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_key.status == STARTED and not waitOnFlip:
            theseKeys = end_key.getKeys(keyList=None, ignoreKeys=["escape"], waitRelease=False)
            _end_key_allKeys.extend(theseKeys)
            if len(_end_key_allKeys):
                end_key.keys = _end_key_allKeys[-1].name  # just the last key pressed
                end_key.rt = _end_key_allKeys[-1].rt
                end_key.duration = _end_key_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=end,
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # check responses
    if end_key.keys in ['', [], None]:  # No response was made
        end_key.keys = None
    thisExp.addData('end_key.keys',end_key.keys)
    if end_key.keys != None:  # we had a response
        thisExp.addData('end_key.rt', end_key.rt)
        thisExp.addData('end_key.duration', end_key.duration)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
