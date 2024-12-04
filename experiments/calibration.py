import torch
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

color = ['b', 'r', 'g', 'c', 'y', 'orange', 'orange']
font_text = {'family' : 'serif',
'weight' : 'normal', 'color':'black',
'size'  : 15,
}
legend_text = {
'size'  : 18,
}

label_text  = {'family' : 'serif',
'weight' : 'normal', 'color':'black',
'size'  : 30,
}

title_text  = {'family' : 'serif',
'weight' : 'normal', 'color':'black',
'size'  : 30,
}


def get_calibration(probs, preds, gts): # input is tensor
    # refer: https://pureai.com/articles/2021/03/03/ml-calibration.aspx
    # 这里运算久一点，可以先写好结果
    accuracy = [0 for _ in range(10)]
    calibration = [0 for _ in range(10)]
    errors = 0
    bins = [[0.0, 0.1], [0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6], [0.6, 0.7], [0.7, 0.8],
           [0.8, 0.9], [0.9, 1.0]]
    errors_correct, errors_incorrect = 0, 0
    n_correct, n_incorrect = 0, 0

    probs, preds, gts = (torch.flatten(probs).detach().numpy(), torch.flatten(preds).detach().numpy(),
                         torch.flatten(gts).numpy())

    assert len(probs) == len(preds) == len(gts), '{}!={}'.format(probs.shape, gts.shape)

    for i in range(len(accuracy)):
        indices = np.where((probs > bins[i][0]) & (probs <= bins[i][1]))
        if len(indices[0]) != 0:
            prob_, pred_, gt_ = probs[indices], preds[indices], gts[indices]
            mean_prob = np.mean(prob_)
            acc_ = np.equal(pred_, gt_).sum() / len(indices[0])
            diff = np.abs(acc_ - mean_prob) # accuracy - average predict
            counts = diff * len(indices[0])
            accuracy[i] = acc_
            calibration[i] = mean_prob
            errors += counts

            # analysis mis-prediction
            acc_indices = np.equal(pred_, gt_)
            prob_, pred_, gt_ = probs[indices][acc_indices==1], preds[indices][acc_indices==1], gts[indices][acc_indices==1]
            errors_correct += np.nan_to_num(np.abs(1 - np.mean(prob_))) * len(prob_)
            n_correct += len(prob_)
            prob_, pred_, gt_ = (probs[indices][acc_indices == 0], preds[indices][acc_indices == 0],
                                 gts[indices][acc_indices == 0])
            errors_incorrect += np.nan_to_num(np.abs(- np.mean(prob_))) * len(prob_)
            n_incorrect += len(prob_)
    errors = errors / len(probs)
    print([float('{:.3f}'.format(i)) for i in accuracy])
    print([float('{:.3f}'.format(i)) for i in calibration])
    print(errors, errors_correct / n_correct, errors_incorrect / n_incorrect)


class Calibration:
    def __init__(self):
        # 这里我用的ACDC-fog
        self.source = [[0.0, 0.051, 0.064, 0.106, 0.157, 0.427, 0.442, 0.458, 0.485, 0.799],
                       [0.0, 0.187, 0.272, 0.362, 0.459, 0.55, 0.651, 0.752, 0.854, 0.991],
                       [0.2062412835026101, 0.03517036225031381, 0.8764278779419856]]
        self.tent = [[0.0, 0.058, 0.16, 0.246, 0.322, 0.409, 0.448, 0.5, 0.559, 0.851],
                    [0.0, 0.185, 0.272, 0.363, 0.459, 0.55, 0.651, 0.752, 0.854, 0.99],
                    [0.15786139979843675, 0.043266101381597054, 0.8284464840189095]]
        # 0.025和0.097在第一个区间可能是有的
        self.svdp = [[0.025, 0.069, 0.102, 0.157, 0.242, 0.459, 0.55, 0.721, 0.788, 0.925],
                    [0.097, 0.184, 0.271, 0.358, 0.456, 0.551, 0.653, 0.749, 0.852, 0.989],
                    [0.07049080722568857, 0.07731656267680875, 0.7234383341704087]]
        self.ours = [[0.0, 0.06, 0.091, 0.159, 0.236, 0.367, 0.434, 0.509, 0.602, 0.939],
                     [0.0, 0.183, 0.268, 0.359, 0.457, 0.55, 0.65, 0.752, 0.854, 0.995],
                     [0.07777708991759666, 0.021905406446223335, 0.7885158050011493]]

    def get_data(self, path):
        state = torch.load(path)
        gts, probs, preds = state['gts'].long(), state['probs'].float(), state['preds'].long()
        print(gts.shape, probs.shape, preds.shape)
        get_calibration(probs=probs, preds=preds, gts=gts)

    def make_model_diagrams(self, n_bins=10):
        # path = './Run_tta_exp/CalibrationExP/fog/analysis.pkl'
        # self.get_data(path)
        titles = ['(a) Source', '(b) TENT', '(c) SVDP', '(d) Ours']
        # svdp虽然miou没有ours高，但是他的diagrams更好，调换一下看看效果
        # data = [self.source, self.tent, self.svdp, self.ours]
        data = [self.source, self.tent, self.ours, self.svdp]

        with PdfPages(r'./calibration.pdf') as pdf:
            bbox_props = dict(boxstyle="square", fc="lightgrey", ec="gray", lw=1.5)
            width = 1.0 / n_bins
            bin_centers = np.linspace(0, 1.0 - width, n_bins) + width / 2
            fig, axs = plt.subplots(1, 4, figsize=(24, 6))
            # plt.rcParams['axes.facecolor'] = 'snow'
            for i in range(len(data)):
                axs[i].set_facecolor("gainsboro")
                confs = axs[i].bar(bin_centers, data[i][0], color='dodgerblue', width=width, ec='black')
                gaps = axs[i].bar(bin_centers, np.asarray(data[i][1]) - np.asarray(data[i][0]), bottom=data[i][0],
                                  color='royalblue', alpha=0.5, width=width, hatch='//', edgecolor='blue')
                axs[i].legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')
                # Clean up
                axs[i].text(0.27, 0.72, "ECE: {:.2f}%".format(data[i][2][0]*100), ha="center", va="center", size=20,
                            weight='normal',
                            bbox=bbox_props)

                # confs = axs[1].bar(bin_centers, data[1][0], color=[0, 0, 1], width=width, ec='black')
                # gaps = axs[1].bar(bin_centers, np.asarray(data[1][1])-np.asarray(data[1][0]), bottom=data[1][0],
                #                color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
                # axs[1].plot([1, 1], [0, 1], '--', color='gray')
                # axs[1].legend([confs, gaps], ['Accuracy', 'Gap'], loc='upper left', fontsize='x-large')
                # axs[1].text(0.17, 0.82, "ECE: {}".format(data[1][2]), ha="center", va="center", size=20, weight='normal',
                #             bbox=bbox_props)
                #
                # axs[0].set_title("Reliability Diagram (SO)", size=22)
                # axs[1].set_title("Reliability Diagram (TENT)", size=22)

            for j, ax in enumerate(axs.flat):
                ax.set_ylabel("Accuracy", fontdict=label_text)
                ax.set_xlabel("Confidence", fontdict=label_text)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
                ax.plot([0, 1], [0, 1], '--', color='gray')
                ax.set_title("{}".format(titles[j]), fontdict=title_text)
                # ax.set_facecolor("snow")

            plt.tight_layout()
            pdf.savefig()
            # plt.show()
            plt.close()


if __name__ == '__main__':
    ece = Calibration()
    ece.make_model_diagrams()