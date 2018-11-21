import time
import numpy as np
from NeuralNet import NeuralNet
import requests
import re

"""
Name : getData.py
Authors : Viktor-Adam Koropecky, David Zahour

Algorithm to test our basic Neural Network 

"""


print("Algorithm Started")

while True:
	print("Requesting data.")
	x = requests.get("http://dvtech.tk:9001/python").text
	y = [int(s) for s in x.split() if s.isdigit()]

	y = re.findall(r'\d+',x)

	ir      = y[7]
	red     = y[8]
	temp    = y[9]
	voltage = y[10]

	print("Data received: ")
	print("Temp value " + temp)
	print("Red value " + red)
	print("IR value " + ir)
	print("Voltage " + voltage)
	print()
	
	nn = NeuralNet()
	print("Utilizing pre-learned NeuralNet located in \"network.xml\".")
	nn.import_network()
	
	inputs = np.array([temp, red, ir], dtype=float)
	inputs = inputs/65536
	out = nn.translate_data(inputs)*65536

	print("Real voltage raw data: ", voltage)
	print("Guessed voltage raw data: ", out)
	print()
	time.sleep(5)
