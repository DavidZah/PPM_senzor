import numpy as np
import random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import json



class NeuralNet:
	"""
	Simple Neural Network that serves as a prototype for real data estimation.
	"""
	def __init__(self):
		np.random.seed(1)
		self.links0 = 2*np.random.random((3,4))-1
		self.links1 = 2*np.random.random((4,1))-1

	def __sigmoid_function(self, x, derivative=False):
		if derivative == True:
			return x*(1-x)

		return 1 / (1 + np.exp(-x))

	# Back propagation is done via Gradient Descent
	def train(self, training_set_inputs, training_set_outputs, iterations):
		for j in range(60000):

			nodes0 = training_set_inputs
			nodes1 = self.__sigmoid_function(np.dot(nodes0, self.links0))
			nodes2 = self.__sigmoid_function(np.dot(nodes1, self.links1))

			nodes2_error = training_set_outputs - nodes2
			nodes2_delta = nodes2_error*self.__sigmoid_function(nodes2, derivative=True)

			nodes1_error = nodes2_delta.dot(self.links1.T)
			nodes1_delta = nodes1_error * self.__sigmoid_function(nodes1, derivative=True)

			self.links1 += nodes1.T.dot(nodes2_delta)
			self.links0 += nodes0.T.dot(nodes1_delta)

		print("Trained to output values of: " , nodes2*65536)

	def translate_data(self, inputs):
		nodes0 = inputs
		nodes1 = self.__sigmoid_function(np.dot(nodes0, self.links0))
		nodes2 = self.__sigmoid_function(np.dot(nodes1, self.links1))
		return nodes2

	def export_network(self, filename="network.xml", path=""):
		file_path = Path(path + filename)
		file = open(file_path, "wb+")
			
		net = ET.Element('Network')
		syn0 = ET.SubElement(net, 'syn0')
		syn0.text = json.dumps(self.links0.tolist())
		syn1 = ET.SubElement(net, 'syn1')
		syn1.text = json.dumps(self.links1.tolist())
		
		file.write(ET.tostring(net, 'utf-8'))
		file.close()

	def import_network(self, filename="network.xml", path=""):
		file_path = Path(path + filename)
		if not file_path.is_file():
			print("Requested network not found.")
			return 0
		tree = ET.parse(file_path)
		net = tree.getroot()
		self.links0 = np.array(json.loads(net.find('syn0').text))
		self.links1 = np.array(json.loads(net.find('syn1').text))