# -*- coding: utf-8 -*-
"""Test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EPel-mbccuunMgq0eFmpZySHvkDAsF3V
"""

import subprocess
requirements_path = #Enter the path to the requirements file here #input("What is the path to the requirements text file: ")
subprocess.check_call(['pip', 'install', '-r', requirements_path])

subprocess.run(['python', '<path to evaluate.py', '-c', '/content/Cam2BEV/model/config.1_FRLR.unetxst.yml', '--model-weights', '<path to the saved model weight'])