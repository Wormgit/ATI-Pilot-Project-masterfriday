#!/usr/bin/env python

# Core libraries
import os
import sys
sys.path.append("../")

# My libraries
from config import cfg
from Utilities.Utility import Utility
from Utilities.DataUtils import DataUtils
from Identifier.MetricLearningWrapper import MetricLearningWrapper
# from Identifier.ClosedSetWrapper import ClosedSetWrapper

class IdentificationManager(object):
	"""
	This class manages the identification pipeline
	"""
	
	def __init__(self, load_weights=True):
		"""
		Class constructor
		"""

		# Initialise our identification method
		if cfg.ID.ID_METHOD == cfg.IDENTIFIERS.METRIC_LEARNING:
			self.__identifier = MetricLearningWrapper()
		elif cfg.ID.ID_METHOD == cfg.IDENTIFIERS.CLOSED_SET:
			self.__identifier = ClosedSetWrapper()

	def identifyBatch(self, images):
		return self.__identifier.predictBatch(images)

	def identifySingle(self, image):
		"""
		Use the currently selected identification technique to identify a given image
		"""
		ID = self.__identifier.predictSingle(image)
		return ID


# Entry method/unit testing method
if __name__ == '__main__':
	pass
