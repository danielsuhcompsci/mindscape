import os
import foldrpp
from collections import defaultdict
import regex as re

    
person_model = foldrpp.load_model_from_file('../fold-rpp-models/person_model_final_fr.txt')

print("Person model ruleset: ")
# print ruleset
for r in person_model.asp():
    print(r)
print()

backpack_model = foldrpp.load_model_from_file('../fold-rpp-models/backpack_model_final_fr.txt')

def is_valid_model(model: foldrpp.Foldrpp):
    return len(model.asp() ) > 0

def get_valid_models():
    for file in os.listdir('../fold-rpp-models'):
        if file.endswith('.txt'):
            model = foldrpp.load_model_from_file('../fold-rpp-models/' + file)
            if is_valid_model(model):
                yield model
            
valid_models = list(get_valid_models())

print("hi")
# Regex pattern to match voxel names
pattern = re.compile(r'\b([a-z0-9]+(?:_?\d*)-\d+)\b')

# Dictionary to count occurrences
frequency_dict = defaultdict(int)    

model_to_unique_voxel_counts = {}

for model in valid_models:
        model_frequency_dict = defaultdict(int)
        print()
        i = 0
        model_name = ""
        for r in model.asp():
            if i == 0:
                # all characters up to the first '('
                model_name = r[:r.find('(')]
                
            i += 1
            print(r)
            
            # Parse and count
            matches = pattern.findall(r)
            for match in matches:
                frequency_dict[match] += 1  
                model_frequency_dict[match] += 1
            # print(f"matches: {matches} for rule: {r}")                                                              
        print()        
        print(f"Model uses {len(model_frequency_dict)} unique voxels")
        print()  
        
        model_to_unique_voxel_counts[model_name] = len(model_frequency_dict)

top_k = 15

# Sort the dictionary by value
sorted_dict = sorted(frequency_dict.items(), key=lambda x: x[1], reverse=True)

# Print the top k
for i, (voxel, count) in enumerate(sorted_dict):
    if i == top_k:
        break
    print(f"{voxel}: {count}")

# print models to unique voxel counts
print()
print("Models to unique voxel counts")
for model, count in model_to_unique_voxel_counts.items():
    print(f"{model}: {count}")

print(f"Total number of unique voxels: {len(frequency_dict)}")

print(f"Average number of unique voxels per model: {sum(model_to_unique_voxel_counts.values()) / len(model_to_unique_voxel_counts)}")

print(f"There are {len(valid_models)} valid models")