from foldrpp import split_data, get_scores, binary_only
from datasets import *
from brainDataset import brainVoxels

categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                  "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]




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



def do_cat(category, truncate=None):
    print('Category:', category)
    model, data = brainVoxels(category, '../FOLDdata/subj01New.csv', 5277, True, truncate = truncate)


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
    save_model_to_file(model, category+'_model_final_fr.txt')
    saved_model = load_model_from_file(category+'_model_final_fr.txt')
        
    ys_test_hat = saved_model.predict(data_test)
    ys_test = [x['label'] for x in data_test]
    stats = get_scores(ys_test_hat, ys_test)
    
    return stats
    
if __name__ == '__main__':

    import time

    profile = False
    num_cores = 8


    if num_cores is None:
        if profile:
            import cProfile
            profiler = cProfile.Profile()
            profiler.enable()
        t = time.time()

        for cat in categories:
            stats = do_cat(cat)
            print(stats)
        print('Time:', time.time() - t)
        if profile:
            profiler.disable()
            profiler.dump_stats('profile_results')

    else:
        import multiprocessing

        #lock for json file
        json_lock = multiprocessing.Lock()

        def worker(i):
            for cat in categories[i::num_cores]:
                stats = do_cat(cat)
                with json_lock:
                    add_stats(cat, *stats)
        
        processes = [multiprocessing.Process(target=worker, args=(i,)) for i in range(num_cores)]
        
        start = time.time()

        for p in processes:
            p.start()
        
        for p in processes:
            p.join()
        
        print('Time:', time.time() - start)