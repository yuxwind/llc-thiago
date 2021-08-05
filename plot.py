import os

import numpy as np
import tensorflow as tf
import pickle


COLORS = ['red', 'orange', 'gold','pink',  'green','lightseagreen', 'cyan', 'blue', 'violet', 'gray', 'brown', 'magenta', 'lightsteelblue' ]

# Plot the number of regions over the square of the number of neurons as a function of epoch.
for n, network in enumerate(NETWORKS):
    results = all_results[str(network)][:REPEATS]
    print(str(network))
    counts = []
    for result in results:
        counts.append(np.mean(np.array(result['counts']), axis=1))
    #mean = np.mean(np.array(counts), axis=0) / (np.sum(network) ** 2)
    #std = np.std(np.array(counts), axis=0) / (np.sum(network) ** 2)
    mean = np.mean(np.array(counts), axis=0)
    std = np.std(np.array(counts), axis=0)
    plt.plot(REPORT_EPOCHS[:NUM_REPORTS], mean[:NUM_REPORTS], label=str(network),
             c=COLORS[n], marker='.')
    plt.fill_between(REPORT_EPOCHS[:NUM_REPORTS],
                     mean[:NUM_REPORTS] - std[:NUM_REPORTS],
                     mean[:NUM_REPORTS] + std[:NUM_REPORTS],
                     color=COLORS[n], alpha=0.1)

plt.xlabel('Epoch', size=20)
#plt.ylabel('Number of regions over\nsquared number of neurons', size=20)
plt.ylabel('Number of regions', size=20)
#plt.show()
plt.legend(loc='upper left', bbox_to_anchor=(1.04, 1))
plt.savefig(os.path.join(fig_dir, 'count_mnist_2d_1.pdf'), bbox_inches='tight')
plt.clf()
