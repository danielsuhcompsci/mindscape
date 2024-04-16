from foldrm import *
from brainDataset import brainVoxels
from timeit import default_timer as timer
from datetime import timedelta

#code originally from https://github.com/hwd404/FOLD-RM/tree/main
def main(): #minor modifications made for Mindscape purposes by Isaac Philo; the CSV used here was generated by Daniel Herrera
    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "x-coord", "y-coord"]

    for category in categories:
        print(f"Beginning analysis on category {category}")
        model, data = brainVoxels(category, '..\..\mindscape\FOLDdata\subj01.csv')
        data_train, data_test = split_data(data, ratio=0.8)

        start = timer()
        model.fit(data_train, ratio=0.5)
        end = timer()

        model.print_asp(simple=True)
        Y = [d[-1] for d in data_test]
        Y_test_hat = model.predict(data_test)
        acc = get_scores(Y_test_hat, data_test)
        print('% acc', round(acc, 4), '# rules', len(model.crs))
        acc, p, r, f1 = scores(Y_test_hat, Y, weighted=True)
        print('% acc', round(acc, 4), 'macro p r f1', round(p, 4), round(r, 4), round(f1, 4), '# rules', len(model.crs))
        print('% foldrm costs: ', timedelta(seconds=end - start), '\n')

    # k = 1
    # for i in range(len(data_test)):
    #     print('Explanation for example number', k, ':')
    #     print(model.explain(data_test[i]))
    #     print('Proof Tree for example number', k, ':')
    #     print(model.proof(data_test[i]))
    #     k += 1


if __name__ == '__main__':
    main()