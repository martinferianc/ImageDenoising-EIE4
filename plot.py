import numpy as np
import matplotlib.pyplot as plt
import glob
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


def plot_results(results, models,model_names, type="Matching"):

    # Select which depths are unique and build the dictionaries
    unique_depths = [m[2] for m in models]
    unique_depths = list(set(unique_depths))

    filters = {}
    n_nodes = {}
    depth_results = {}
    for depth in unique_depths:
        for i,m in enumerate(models):

            # Do an encoding for each depth
            if depth == m[2]:

                f = int(m[0])
                if f == 32:
                    f = 1
                elif f == 64:
                    f = 2
                elif f == 128:
                    f = 3

                n = int(''.join(m[1]))
                if n == 135:
                    n = 1
                elif n == 359:
                    n = 2
                elif n == 5915:
                    n = 3
                if depth in filters:
                    filters[depth].append(f)
                    n_nodes[depth].append(n)
                    depth_results[depth].append(float(results[i]))
                else:
                    filters[depth] = [f]
                    n_nodes[depth] = [n]
                    depth_results[depth] = [float(results[i])]

    fig = plt.figure(figsize=(12,3))
    fig.tight_layout()

    # Do the plots
    for i,depth in enumerate(unique_depths):
        ax = fig.add_subplot(1, len(unique_depths), i+1, projection='3d')
        ax.set_xticks([1,2,3])
        ax.set_xticklabels([32,64,128])
        ax.set_yticks([1,2,3])
        ax.set_yticklabels(["1,3,5", "3,5,9", "5,9,15"])
        ax.scatter(filters[depth], n_nodes[depth], depth_results[depth], depthshade=True, s=40)
        ax.set_xlabel('Denoise Filters', linespacing=3.2)
        ax.set_ylabel('Denoise $N$ Nodes', linespacing=3.1)
        ax.set_zlabel('mAP', linespacing=3.4)
        ax.grid(True)
        ax.set_title("Results for fixed \n descriptor depth = {}".format(depth))

    fig.savefig("Figures/{}_results.png".format(type))

    # Return the best performer for all depths
    best_performer = [x for _,x in sorted(zip(results,model_names))][-1]
    print("The best result for {} was achieved with: {}".format(type, best_performer))


def load_file(path):
    retrieval_results = []
    matching_results = []
    verification_results = []
    models = []
    model_names = []

    for filename in glob.glob(path+'*.txt'):
        model_names.append(filename)
        with open(filename) as f:
            # Here is the entire content of the data file
            content = f.readlines()
            content = [x.strip() for x in content]

            # Organisation of the file
            # 1. Line model type
            # 2. Line Mean Retrieval Results
            # 3. Line Mean Matching Results
            # 4. - 10. Line Verification Results

            # Model type
            c = content[0].strip().split()
            models.append([c[0], c[1:4], c[4]])

            # Mean Retrieval Results
            m = content[1].strip().split()
            retrieval_results.append(np.mean([float(i) for i in m]))

            # Mean Verification Results
            v = content[2].strip().split()
            verification_results.append(np.mean([float(i) for i in v]))

            # Mean Matching Results
            balanced = []
            imbalanced = []
            for i in range(3,len(content)):
                line = content[i].strip().split()
                if i-3 <=2:
                    balanced.append(np.mean([float(i) for i in line]))
                else:
                    imbalanced.append(np.mean([float(i) for i in line]))

            matching_results.append(np.mean([np.mean(balanced), np.mean(imbalanced)]))

    return  model_names, models, retrieval_results, verification_results, matching_results

if __name__ == '__main__':
    model_names, models, retrieval_results, verification_results, matching_results = load_file("Data/")

    plot_results(retrieval_results, models,model_names, "Retrieval")
    plot_results(verification_results, models,model_names, "Verification")
    plot_results(matching_results, models,model_names, "Matching")
