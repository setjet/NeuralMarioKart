import os
import sys
from multiprocessing import Process
from setup.util import run_server, XboxController, NeuralNetwork

def run_emulator():
  os.system("cd n64/test; ./mupen64plus --input ../source/mupen64plus-input-bot/mupen64plus-input-bot.so MarioKart64.n64")

def run_communicator():
  if len(sys.argv) > 1 and sys.argv[1] in ['NN','AI']:
    run_server(NeuralNetwork())
  else:
    run_server(XboxController())

p1 = Process(target=run_emulator)
p2 = Process(target=run_communicator)
p1.start()
p2.start()
