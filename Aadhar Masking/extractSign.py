import numpy as np
import cv2
import os
import requests


r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
print(r)
