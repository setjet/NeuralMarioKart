import os
import sys
from multiprocessing import Process
from util import run_server, xboxController, neuralNetwork

def run_emulator():
  os.system("cd n64/test; ./mupen64plus --input ../source/mupen64plus-input-bot/mupen64plus-input-bot.so MarioKart64.n64")

def run_communicator():
  if len(sys.argv) > 1 and sys.argv[1] in ['NN','AI','NeuralNetwork','Neuralnetwork']:
    run_server(neuralNetwork())
  else:
    print "whoop"
    run_server(xboxController())

p1 = Process(target=run_emulator)
p2 = Process(target=run_communicator)
p1.start()
p2.start()
