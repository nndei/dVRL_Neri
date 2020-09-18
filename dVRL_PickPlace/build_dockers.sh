#!/bin/bash

cd dVRL_simulator/environments/
cd pick_ee_dockerfile/ && docker build -t vrep_ee_pickplace .
