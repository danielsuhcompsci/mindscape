import torch
import numpy as np
import matplotlib.pyplot as plt


num_trials = 5

allmetrics =[]


# Load all the metrics
for i in range(num_trials):
    filename = f"./metrics/trial_{i+1}_metrics.pth"
    allmetrics.append(torch.load(filename, map_location=torch.device('cpu')))
    allmetrics[i]['r2s'] = [x.numpy() for x in allmetrics[i]['r2s']]


def gen_graphs(prefix = None):

    if prefix is None:
        prefix = ""
    prefix = f"./graphs/{prefix}"

    
    # Make graph for each trial
    for trial, metrics in enumerate(allmetrics):

        r2s = metrics['r2s']
        losses = metrics['losses']

        print(len(r2s), " ",  len(losses))

        median_r2s = [np.median(r2) for r2 in r2s]
        train_losses = [x[0] for x in losses]
        val_losses = [x[1] for x in losses]
        
        max_loss = max(train_losses + val_losses)

        #plot median r2, train loss, val loss
        #ornage is train, green is val, blue is r2
        plt.figure()
        plt.ylim(bottom=0, top = max_loss*1.1)
        # plt.plot(median_r2s, color='blue')
        plt.plot(train_losses, color='orange')
        plt.plot(val_losses, color='blue')
        plt.title(f"Trial {trial+1}")

        # horizontal axis is epochs
        plt.xlabel("Epochs")
        plt.ylabel("MSE Loss")

        #add legend
        plt.legend(['Train Loss', 'Val Loss'])


        #show and wait for input

        #saev png

        plt.savefig(f"{prefix}trial_{trial+1}.png")        


    #for these, trials are on same graph. colors are:
    # trial 1: blue, trial 2: orange, trial 3: green, trial 4: red, trial 5: purple

    # Make graph for all val losses

    all_val_losses = []
    for trial, metrics in enumerate(allmetrics):
        val_losses = [x[1] for x in metrics['losses']]
        all_val_losses += val_losses
    
    max_val_loss = max(all_val_losses)

    colors = ['blue', 'orange', 'green', 'red', 'purple']
    plt.figure()
    plt.ylim(bottom=0, top = max_loss*1.1)
    plt.title("Validation Losses")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")

    for trial, metrics in enumerate(allmetrics):
        val_losses = [x[1] for x in metrics['losses']]
        plt.plot(val_losses, color=colors[trial])

    #legend
    plt.legend([f'Trial {i+1}' for i in range(num_trials)])

    #save_png
    plt.savefig(f"{prefix}val_losses.png")


    #graph with all median r2s, same colors
    plt.figure()
    
    # all median r2s
    meds = [np.median(r2) for r2 in allmetrics[0]['r2s']]
    low_med = min(meds)

    bottom = min(0,low_med - 0.1)
    plt.ylim(bottom=bottom, top = None)
    plt.title("Median R2s")
    plt.xlabel("Epochs")
    plt.ylabel("R2")

    for trial, metrics in enumerate(allmetrics):
        r2s = metrics['r2s']
        median_r2s = [np.median(r2) for r2 in r2s]
        plt.plot(median_r2s, color=colors[trial])

    #legend
    plt.legend([f'Trial {i+1}' for i in range(num_trials)])

    #save_png
    plt.savefig(f"{prefix}median_r2s.png")


gen_graphs()

for i, m, in enumerate(allmetrics):
    #truncate everything to 30 epochs
    allmetrics[i]['r2s'] = m['r2s'][:30]
    allmetrics[i]['losses'] = m['losses'][:30]

gen_graphs("truncated_")