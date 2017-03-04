#!/bin/sh
cd n64
./m64p_get.sh && ./m64p_build.sh && ./m64p_test.sh
mv MarioKart64.n64 test/MarioKart64.n64
echo "************************************ Running test, close if working properly"
cd source
echo "************************************ Downloading emulator input plugin"
git clone https://github.com/kevinhughes27/mupen64plus-input-bot
cd mupen64plus-input-bot
echo "************************************ Making emulator input plugin"
make all
