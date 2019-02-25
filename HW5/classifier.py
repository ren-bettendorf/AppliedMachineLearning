import numpy as np
import matplotlib.pyplot as plt
import glob

CLASSIFIER_LABELS = ["Brush_Teeth", "Climb_Stairs", "Comb_Hair", "Descend_Stairs",
                     "Drink_Glass", "Eat_Meat", "Eat_Soup", "Getup_Bed", "Liedown_Bed",
                     "Pour_Water", "Sitdown_Chair", "Standup_Chair", "Use_Telephone", "Walk"]

for label in CLASSIFIER_LABELS:
    file_list = glob.glob("HMP_Dataset/" + label + "/*.txt")
    for file in file_list:
        data = np.genfromtxt(file, delimiter=" ")
        print(f"Reading {file} with \n\n{data}")
