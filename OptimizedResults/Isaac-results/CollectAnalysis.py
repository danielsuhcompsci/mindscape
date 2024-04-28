import os
import re
import matplotlib.pyplot as plt

accuracyRegex = re.compile(r'acc (0\.[0-9]{3}).*') #regex for the accuracy line
categoryRegex = re.compile('(.+)_Optimized') #match everything before the first underscore
statsRegex = re.compile('.*Stats.*')
categories = []
accuracies = []
for file_name in os.listdir(os.getcwd()):
    if os.path.isfile(os.path.join(os.getcwd(), file_name)):
        if statsRegex.match(str(file_name)):
            file = open(os.path.join(os.getcwd(), file_name), 'r')
            print(file_name)
            categoryName = categoryRegex.match(file_name).group(1)  # get the category name
            for index, line in enumerate(file):
                if "acc" in line:
                    print(line)
                    accuracy = float(line[6:11])  # get the accuracy
                    categories.append(categoryName)
                    accuracies.append(accuracy)
                    print("Inner loop")
                    break
bars = plt.bar(categories, accuracies, color='green')
xticks = [position for position in range(80)]
plt.xticks(xticks, categories, fontsize='6', rotation=90)
plt.subplots_adjust(bottom=0.2)
# plt.rcParams['font.size'] = 1
plt.xlabel('Category')
plt.ylabel('Accuracy')
plt.title('FOLD-RPP Prediction Accuracy per Subject')
plt.style.use("_mpl-gallery")
usegreen = True
for i in range(80):
    if(usegreen):
        bars[i].set_color("cyan")
        usegreen = not usegreen
    else:
        bars[i].set_color('black')
        usegreen = not usegreen
plt.show()
print(categories, accuracies)