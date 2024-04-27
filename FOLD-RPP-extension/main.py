from foldrpp import split_data, get_scores, num_predicates, binary_only
from datasets import *
from brainDataset import brainVoxels
from timeit import default_timer as timer
from datetime import timedelta
import pstats
import cProfile

def main():
    # categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    #               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    #               "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
    #               "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    #               "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    #               "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    #               "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    #               "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    #               "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "x-coord", "y-coord"]
    categories = [ "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "pizza", "donut",
                  "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    for category in categories:
        profiler = cProfile.Profile()
        print(f"Beginning analysis on category {category}")
        load_start = timer()
        profiler.enable()
        model, data = brainVoxels(category, '../FOLDdata/subj01New.csv', 5277, True)
        profiler.disable()
        load_end = timer()
        print('% load data costs: ', timedelta(seconds=load_end - load_start), '\n')

        data_train, data_test = split_data(data, ratio=0.8, rand=True)

        start = timer()
        with binary_only():
            profiler.enable()
            model.fit(data_train)
            profiler.disable()
        end = timer()

        for r in model.asp():
            print(r)

        ys_test_hat = model.predict(data_test)
        ys_test = [x['label'] for x in data_test]
        acc, p, r, f1 = get_scores(ys_test_hat, ys_test)
        print('% acc', round(acc, 3), 'p', round(p, 3), 'r', round(r, 3), 'f1', round(f1, 3))
        n_rules, n_preds = len(model.flat_rules), num_predicates(model.flat_rules)
        print('% #rules', n_rules, '#preds', n_preds)
        print('% foldrpp costs: ', timedelta(seconds=end - start), '\n')

        for x in data_test[:10]:
            for r in model.proof_rules(x):
                print(r)
            for r in model.proof_trees(x):
                print(r)

        from foldrpp import save_model_to_file, load_model_from_file
        save_model_to_file(model, category+'Full.txt')
        saved_model = load_model_from_file(category+'Full.txt')

        ys_test_hat = saved_model.predict(data_test)
        ys_test = [x['label'] for x in data_test]
        acc, p, r, f1 = get_scores(ys_test_hat, ys_test)
        print('% acc', round(acc, 3), 'p', round(p, 3), 'r', round(r, 3), 'f1', round(f1, 3))

            # for x in data_test[:10]:
            #     for r in saved_model.proof_rules(x):
            #         print(r)
            #     for r in saved_model.proof_trees(x):
            #         print(r)

        pstats.Stats(profiler).dump_stats('profile_data.prof')

        break
if __name__ == '__main__':
    main()
