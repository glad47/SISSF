

import logging as lg

log = lg.getLogger('different')
log.setLevel(lg.DEBUG)  # Set the desired logging level

# Create a FileHandler to write logs to a file
file_handler2 = lg.FileHandler('different.log')
file_handler2.setLevel(lg.DEBUG)  # Set the desired logging level for the file

# Create a formatter for the log messages
formatter = lg.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler2.setFormatter(formatter)

# Add the FileHandler to the logger
log.addHandler(file_handler2)