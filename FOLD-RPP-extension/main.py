import datetime

from foldrpp import split_data, get_scores, num_predicates, binary_only
from datasets import *
from brainDataset import brainVoxels
from timeit import default_timer as timer
from datetime import timedelta
from brainDataset import get_attrs

profile = True

if profile:
    import pstats
    import cProfile


def main():
    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                  "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
                  "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
                  "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
                  "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                  "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
                  "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "x-coord", "y-coord"]
    # csvs = ['../FOLDdata/subj01New-500.csv']
    csvs = ['../FOLDdata/subj01Trunc.csv', '../FOLDdata/subj01Modified.csv']
    for filename in csvs:
        columns = get_attrs(filename, categories)
        for category in categories:
            if profile:
                profiler = cProfile.Profile()

            chrono = str(datetime.datetime.today())[:10]
            savefilename = (filename.split('/')[-1].removesuffix('.csv')) + '_' + category + '_Optimized_' + chrono
            statsFileName = savefilename + '_Stats.txt'
            savefilename += '.txt'
            statsString = f"Beginning analysis on category {category} in file {filename}"
            print(f"Beginning analysis on category {category} in file {filename}")
            load_start = timer()
            if profile:
                profiler.enable()
            model, data = brainVoxels(category, filename, columns)

            # print(f"data[0] == {data[0]} and data[1] == {data[1]}")

            if profile:
                profiler.disable()
            load_end = timer()

            statsString += '\n' + '% load data costs: ' + str(timedelta(seconds=load_end - load_start)) + '\n'
            print('% load data costs: ', timedelta(seconds=load_end - load_start), '\n')

            data_train, data_test = split_data(data, ratio=0.8, rand=True)

            start = timer()
            with binary_only():
                if profile:
                    profiler.enable()
                model.fit(data_train)
                if profile:
                    profiler.disable()
            end = timer()

            for r in model.asp():
                statsString += "\n" + str(r)
                print(r)

            ys_test_hat = model.predict(data_test)
            ys_test = [x['label'] for x in data_test]
            acc, p, r, f1 = get_scores(ys_test_hat, ys_test)
            statsString += '\n% acc ' + str(round(acc, 3)) + ' p ' + str(round(p, 3)) + ' r ' + str(round(r, 3)) + ' f1 ' + str(round(f1, 3))
            print('% acc', round(acc, 3), 'p', round(p, 3), 'r', round(r, 3), 'f1', round(f1, 3))

            n_rules, n_preds = len(model.flat_rules), num_predicates(model.flat_rules)
            statsString += '\n% #rules ' + str(n_rules) + ' #preds ' + str(n_preds)
            print('% #rules', n_rules, '#preds', n_preds)
            statsString += '\n% foldrpp costs: ' + str(timedelta(seconds=end - start)) + '\n'
            print('% foldrpp costs: ', timedelta(seconds=end - start), '\n')

            for x in data_test[:10]:
                for r in model.proof_rules(x):
                    statsString += '\n' + str(r)
                    print(r)
                for r in model.proof_trees(x):
                    statsString += '\n'+ str(r)
                    print(r)

            from foldrpp import save_model_to_file, load_model_from_file
            save_model_to_file(model, savefilename)
            saved_model = load_model_from_file(savefilename)

            ys_test_hat = saved_model.predict(data_test)
            ys_test = [x['label'] for x in data_test]
            acc, p, r, f1 = get_scores(ys_test_hat, ys_test)
            statsString += '\n' + '% acc ' + str(round(acc, 3)) + ' p ' + str(round(p, 3)) + ' r ' + str(round(r, 3)) + ' f1 ' + str(round(f1, 3))
            print('% acc', round(acc, 3), 'p', round(p, 3), 'r', round(r, 3), 'f1', round(f1, 3))

            statsFile = open(statsFileName, mode='w')
            statsFile.write(statsString)
            statsFile.flush()
            statsFile.close()

            # for x in data_test[:10]:
            #     for r in saved_model.proof_rules(x):
            #         print(r)
            #     for r in saved_model.proof_trees(x):
            #         print(r)

            if profile:
                pstats.Stats(profiler).dump_stats('profile_data.prof')

if __name__ == '__main__':
    main()
