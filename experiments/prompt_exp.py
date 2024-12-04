import numpy as np
import matplotlib.pyplot as plt
import csv
import os
from matplotlib.backends.backend_pdf import PdfPages

color = ['b', 'r', 'g', 'c', 'y', 'orange', 'orange']
font_text = {'family': 'serif',
             'weight': 'normal', 'color': 'black',
             }

legend_text = {
    'size': 23,
}

label_text = {'family': 'serif',
              'weight': 'normal', 'color': 'black',
              'size': 25,
              }

title_text = {'family': 'serif',
              'weight': 'normal', 'color': 'black',
              'size': 18,
              }


class CatastrophicForgetting:
    # 这里是要补充ours的数据就可以画图了。但是最终要看是一个图还是两个图
    def __init__(self):
        # fog night rain snow. ACDC的结果好像不太明显，可能得换个数据集看看
        # ACDC
        self.acdc = ['Fog', 'Night', 'Rain', 'Snow']
        self.source_acdc = [60.87, 69.37, 69.36, 69.26]
        # adapt之后source的结果
        self.source_tent_acdc = [58.26, 66.85, 65.12, 63.74]
        self.test_tent_acdc = [60.7, 38.0, 54.4, 55.3]
        # self.source_ft_pl = [32.13, 34.58, 21.26, 7.4]
        # self.test_ft_pl = [57.02, 21.69, 33.28, 21.92]
        self.source_ours_acdc = [72.13, 74.58, 71.26, 77.4]
        self.test_ours_acdc = [67.02, 61.69, 63.28, 61.92]

        # NTHU. 注意要乘以19再除以16
        self.nthu = ['Rio', 'Rome', 'Taipei', 'Tokyo']
        self.source_nthu = np.asarray([34.18, 35.0, 37.5, 36.1]) * 19 / 16
        # adapt之后source的结果
        self.test_tent_nthu = np.asarray([32.5, 33.7, 31.2, 34.0]) * 19 / 16
        self.source_tent_nthu = np.asarray([33.16, 33.8, 36.0, 32.9]) * 19 / 16
        self.test_ours_nthu = np.asarray([37.7, 37.89, 36.6, 36.38]) * 19 / 16
        self.source_ours_nthu = np.asarray([37.8, 38.64, 39.8, 38.58]) * 19 / 16

        # self.source_ft_pl = [32.13, 34.58, 21.26, 7.4]
        # self.test_ft_pl = [57.02, 21.69, 33.28, 21.92]
        # self.source_ours = [72.13, 74.58, 71.26, 77.4]
        # self.test_ours = [67.02, 61.69, 63.28, 61.92]

    def plot_each_dataset(self, axs, data, domains, title):
        markersize = 14
        linewidth = 2.5
        l1 = axs.plot(range(4), data[0], label='Original (ID)', marker='^', markersize=markersize,
                      linestyle='solid', color='red', linewidth=linewidth, markerfacecolor='none')
        l2 = axs.plot(range(4), data[1], label='Tent (ID)', marker='o', markersize=markersize,
                      linestyle='--', color='red', linewidth=linewidth, markerfacecolor='none')
        l3 = axs.plot(range(4), data[2], label='Ours (ID)', marker='*', markersize=markersize,
                      linestyle='solid', color='red', linewidth=linewidth, markerfacecolor='none')

        axs1 = axs.twinx()
        l4 = axs1.plot(range(4), data[3], label='Tent (OOD)', marker='o', markersize=markersize,
                       linestyle='--', color='blue', linewidth=linewidth, markerfacecolor='none')
        l5 = axs1.plot(range(4), data[4], label='Ours (OOD)', marker='*', markersize=markersize,
                       linestyle='solid', color='blue', linewidth=linewidth, markerfacecolor='none')

        line1, = axs.plot([1], label='Original (ID)', marker='^', markersize=markersize,
                          linestyle='--', color='red', linewidth=linewidth, markerfacecolor='none')
        line2, = axs.plot([1], label='Tent (ID)', marker='o', markersize=markersize,
                          linestyle='--', color='red', linewidth=linewidth, markerfacecolor='none')
        line3, = axs.plot([1], label='Ours (ID)', marker='*', markersize=markersize,
                          linestyle='solid', color='red', linewidth=linewidth, markerfacecolor='none')
        line4, = axs.plot([1], label='Tent (OOD)', marker='o', markersize=markersize,
                          linestyle='--', color='blue', linewidth=linewidth, markerfacecolor='none')
        line5, = axs.plot([1], label='Ours (OOD)', marker='*', markersize=markersize,
                          linestyle='solid', color='blue', linewidth=linewidth, markerfacecolor='none')
        axs.legend(handles=[line1, line2, line3, line4, line5], loc='lower left', prop=legend_text)

        axs.grid()
        axs.set_ylabel('ID mIoU (%)', fontdict={'family': 'serif', 'weight': 'normal', 'color': 'red', 'size': 25})
        axs1.set_ylabel('OOD mIoU (%)', fontdict={'family': 'serif', 'weight': 'normal', 'color': 'blue', 'size': 25})
        axs.tick_params(axis='x', labelsize=25)
        axs.tick_params(axis='y', labelsize=25)
        axs1.tick_params(axis='y', labelsize=25)
        axs.set_xticks(range(4))
        axs.set_xticklabels(domains, fontdict=font_text)
        axs.set_title(title, fontdict={'family': 'serif', 'size': 25}, y=-0.2)

    def plot_img(self, domain='fog'):
        markersize = 8
        linewidth = 2.5
        with PdfPages(r'./{}.pdf'.format('catastrophic_forgetting')) as pdf:
            fig, axs = plt.subplots(1, 2, figsize=(24, 6))  # w, h
            self.plot_each_dataset(
                axs[0],
                [self.source_acdc, self.source_tent_acdc, self.source_ours_acdc, self.test_tent_acdc,
                 self.test_ours_acdc],
                self.acdc, '(a) ACDC dataset'
            )

            self.plot_each_dataset(
                axs[1],
                [self.source_nthu, self.source_tent_nthu, self.source_ours_nthu, self.test_tent_nthu,
                 self.test_ours_nthu],
                self.nthu, '(b) NTHU dataset'
            )

            plt.tight_layout()
            pdf.savefig()
            # plt.show()
            plt.close()


# computation time
class AdaptTime:
    def __init__(self):
        # 这里就和SVDP和CoTTA对比运行时间
        self.domains = ['CS-C', 'ACDC', 'NTHU']
        self.tent_cs = [152, ]
        self.ours = [1120, 240, 70]
        self.cotta = [2119.4, 400, 120]
        self.svdp = [4538.4, 540, 180]

    def plot_bar(self, ax, tick_step=1, group_gap=0.2, bar_grap=0):
        def show_bar_label(rects):
            for rect in rects:
                height = rect.get_height()
                print(rect.get_width(), height)
                ax.text(rect.get_x() + rect.get_width() / 2. - 0.10, 1.01 * height, '%s' % int(height), size=22,
                         family="serif")

        x = np.arange(len(self.domains)) * tick_step  # x每一组柱子的起点，一共是3个sub-domains
        group_nums = 3  # 多少组柱子，一组有三个
        group_width = tick_step - group_gap  # 计算一组的长度
        bar_span = group_width / group_nums  # 一组里面每个柱子的宽度
        bar_width = bar_span - bar_grap  # 是否需要bar_grap，柱子之间是否需要有空隙
        legends = ['Ours', 'CoTTA', 'SVDP']
        colors = ['black', 'red', 'blue']
        datas = [self.ours, self.cotta, self.svdp]
        for i in range(3):  # 一个循环是画一个方法在所有domain上的效果，即一次是画多个间隔的柱子不是一组（相连）的柱子
            x_site = x + i * bar_span
            data = datas[i]
            rects = ax.bar(x_site, data, tick_label=self.domains, label=legends[i], width=bar_width, color=colors[i],
                   alpha=0.9, edgecolor='black')
            show_bar_label(rects)

        # ax.set_ylim(50, 95)
        ax.set_xticks(x + 1 * bar_span)  # 这里控制label显示的位置
        ax.set_xticklabels(self.domains, rotation=0, fontdict=label_text)
        ax.tick_params(axis='y', labelsize=22)
        ax.legend(loc='upper right', prop=legend_text)
        ax.set_ylabel('Computation time (s)', fontdict=label_text)
        ax.grid()

    def plot_figure(self):
        with PdfPages(r'./{}.pdf'.format('computation_time')) as pdf:
            fig, axs = plt.subplots(1, 1, figsize=(12, 7))
            self.plot_bar(axs)
            plt.tight_layout()
            pdf.savefig()
            # plt.show()
            plt.close()


# WMoE不同experts数量对结果的影响
class WeightMixtureOfExperts:
    def __init__(self):
        # 改成与一开始很差，10差不多
        # m_experts=(5, 10, 15, 20, 30)
        self.acdc = {
            'Fog': [72.0, 73, 72.6, 73, 72.9],
            'Night': [42.5, 43.1, 44.9, 42.8, 43.2],
            'Rain': [66.1, 66.5, 66.7, 67.0, 66.9],
            'Snow': [65.8, 66.6, 66.5, 66.7, 66.5],
        }

        self.nthu = {
            'Rio': [42.1, 42.5, 43.0, 43.3, 43.4],
            'Rome': [39.0, 39.2, 39.4, 39.2, 40.3],
            'Taipei': [40.5, 40.9, 41, 41.1, 42.5],
            'Tokyo': [33.3, 37.8, 37.6, 38, 34.8],
        }

    def plot_each_dataset(self, axs, domain_dict, title, m_experts=(5, 10, 15, 20, 30)):

        markersize = 14
        linewidth = 2.5
        markers = ['.', 'o', 'v', '2', 'X']
        colors = ['r', 'g', 'b', 'orange', 'y']
        for i, domain in enumerate(domain_dict.keys()):
            axs.plot(range(len(m_experts)), domain_dict[domain], label=domain, marker=markers[i], markersize=markersize,
                    linestyle='solid', color=colors[i], linewidth=linewidth, markerfacecolor='none')

        axs.legend(loc='lower left', prop=legend_text)
        axs.grid()
        axs.set_ylabel('mIoU (%)', fontdict={'family': 'serif', 'weight': 'normal', 'color': 'black', 'size': 25})
        axs.set_xlabel('T', fontdict={'family': 'serif', 'weight': 'normal', 'color': 'black', 'size': 25})
        axs.tick_params(axis='x', labelsize=25)
        axs.tick_params(axis='y', labelsize=25)
        axs.set_xticks(range(len(m_experts)))
        axs.set_xticklabels(m_experts, fontdict=font_text)

        axs.set_title(title, fontdict={'family': 'serif', 'size': 25}, y=-0.3)

    def plot_img(self, domain='fog'):
        markersize = 8
        linewidth = 2.5
        with PdfPages(r'./{}.pdf'.format('WMoE')) as pdf:
            fig, axs = plt.subplots(1, 2, figsize=(24, 6))  # w, h
            self.plot_each_dataset(
                axs[0],
                self.acdc,
                '(a) ACDC dataset'
            )

            self.plot_each_dataset(
                axs[1],
                self.nthu,
                '(b) NTHU dataset'
            )

            plt.tight_layout()
            pdf.savefig()
            # plt.show()
            plt.close()


if __name__ == '__main__':
    # cf = CatastrophicForgetting()
    # cf.plot_img()

    # com_time = AdaptTime()
    # com_time.plot_figure()

    wmoe = WeightMixtureOfExperts()
    wmoe.plot_img()
