import os
import re
import matplotlib.pyplot as plt

regex = re.compile('^% acc (0\\.\\d+).*') #regex for the accuracy line
categoryRegex = re.compile('(.+)_') #match everything before the first underscore
categories = []
accuracies = []
for file_name in os.listdir(os.getcwd()):
    if os.path.isfile(os.path.join(os.getcwd(), file_name)):
        file = open(os.path.join(os.getcwd(), file_name), 'r')
        text = file.read()
        categoryName = regex.match(file_name).group(1) #get the category name
        accuracy = float(regex.match(text).group(1)) #get the accuracy

        categories.append(categoryName)
        accuracies.append(accuracy)
plt.bar(categories, accuracies, color='green')