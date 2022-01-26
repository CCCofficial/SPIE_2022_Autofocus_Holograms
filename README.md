# SPIE_2022_Autofocus_Holograms
Python code used to analyze the performance of five holographic reconstruction autofocus methods.

createFocusScores.py
Reads raw cropped holograms from a directory, performs 5 focus methods over a range of Z values, saves as a csv file

reconstruction.py
A function called by createFocusScores.py to reconstruct a hologram for a specified Z value and return a complex reconstructed image using the angular spectrum method.
