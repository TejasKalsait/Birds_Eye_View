# -*- coding: utf-8 -*-
"""Train.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1tKq9BNtoaV1_lBH1KZzEDTNLQgj1finE
"""

import subprocess
requirements_path = #Enter the path to the requirements file here #input("What is the path to the requirements text file: ")
subprocess.check_call(['pip', 'install', '-r', requirements_path])

subprocess.run(['python', '<path to train.py', '-c', '/content/Cam2BEV/model/config.1_FRLR.unetxst.yml'])