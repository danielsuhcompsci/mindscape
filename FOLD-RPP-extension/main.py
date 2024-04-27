from foldrpp import split_data, get_scores, binary_only
from datasets import *
from brainDataset import brainVoxels



def main():
    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                  "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    for category in categories:
        do_cat(category)



import json

def add_stats(cat, acc, p , r, f1):
    with open("fold_stats.json") as f:
        current = json.load(f)
    
    current[cat] = {
        'acc': acc,
        'p':p,
        'r':r,
        'f1':f1
    }   

    with open("fold_stats.json", "w") as f:
        json.dump(current, f)

if __name__ == '__main__':
    main()


def do_cat(category):
    print('Category:', category)
    model, data = brainVoxels(category, '../FOLDdata/subj01New.csv', 5277, True)


    data_train, data_test = split_data(data, ratio=0.8, rand=True)

    with binary_only():

        model.fit(data_train)


    for r in model.asp():
        print(r)

    ys_test_hat = model.predict(data_test)
    ys_test = [x['label'] for x in data_test]
    acc, p, r, f1 = get_scores(ys_test_hat, ys_test)

    for x in data_test[:10]:
        for r in model.proof_rules(x):
            print(r)
        for r in model.proof_trees(x):
            print(r)

    from foldrpp import save_model_to_file, load_model_from_file
    save_model_to_file(model, category+'_model_final.txt')
    saved_model = load_model_from_file(category+'_model_final.txt')
        
    ys_test_hat = saved_model.predict(data_test)
    ys_test = [x['label'] for x in data_test]
    acc, p, r, f1 = get_scores(ys_test_hat, ys_test)
    
    add_stats(category, acc, p, r, f1)

    print('% acc', round(acc, 3), 'p', round(p, 3), 'r', round(r, 3), 'f1', round(f1, 3))
