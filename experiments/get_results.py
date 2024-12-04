import os
import csv
import torch
import numpy as np


def get_results(path):
    pass


class CorruptionResult:
    def __init__(self):
        self.noise = ['gaussian_noise', 'shot_noise', 'impulse_noise']
        self.blur = ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur']
        self.weather = ['snow', 'frost',	'fog', 'brightness']
        self.digital = ['contrast', 'elastic_transform',	'pixelate', 'jpeg_compression']
        self.names = ['noise', 'blur', 'weather', 'digital']

    def get_result_cifar10(self, baseline='FedAvg_Test', dataset='cifar10_c', corruption='brightness'):
        path = './Run/resnet18/{}/{}/'.format(baseline, dataset)

        for i, domains in enumerate([self.noise, self.blur, self.weather, self.digital]):
            results = []
            for domain in domains:
                domain_path = os.path.join(path, domain)
                result_pths = [f for f in os.listdir(domain_path) if '.pth' in f]
                result = [torch.load(os.path.join(domain_path, pth))['shift_test_wavg_metric'] for pth in result_pths]
                results.append(np.mean(result))

            print('{} results:'.format(self.names[i]), results)


if __name__ == '__main__':
    result = CorruptionResult()
    result.get_result_cifar10()