import os
from multiprocessing import Process
from util import run_server, xboxController, neuralNetwork

def run_emulator():
  os.system("cd n64/test; ./mupen64plus --input ../source/mupen64plus-input-bot/mupen64plus-input-bot.so MarioKart64.n64")

def run_communicator():
  #run_server(xboxController())
  run_server(neuralNetwork())

p1 = Process(target=run_emulator)
p1.start()
p2 = Process(target=run_communicator)
p2.start()
