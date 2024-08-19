'''
Author: your name
Date: 2024-06-13 14:35:08
LastEditTime: 2024-08-19 12:20:02
LastEditors: your name
Description: In User Settings Edit
FilePath: \SISSF\util\logFile.py
'''
## Author: ggffhh3344@gmail.com Abdulaziz Ahmed
## Date: 2024-06-11 11:15:59
## LastEditTime: 2024-08-18 10:15:14

import logging as lg

log = lg.getLogger('log')
log.setLevel(lg.DEBUG)  # Set the desired logging level

# Create a FileHandler to write logs to a file
file_handler2 = lg.FileHandler('log.log')
file_handler2.setLevel(lg.DEBUG)  # Set the desired logging level for the file

# Create a formatter for the log messages
formatter = lg.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler2.setFormatter(formatter)

# Add the FileHandler to the logger
log.addHandler(file_handler2)