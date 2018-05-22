import numpy as np


class InputProcessor:

    def __init__(self, filename):
        self.filename = filename
        self.initial_values = self.read()
        self.means, self.stds, self.values = self.process()

    def read(self):
        with open(self.filename, 'r') as f:
            lines = map(str.strip, f.readlines())
            lines = map(lambda s: list(map(float, s.split(','))), lines)
            return np.array(list(map(lambda s: np.array(s), lines)))

    def process(self):
        means = []
        stds = []
        processed_lines = []
        for line in self.initial_values.T:
            mean = np.mean(line)
            std = np.std(line)
            means.append(mean)
            stds.append(std)

            processed_lines.append(list(map(lambda v: (v - mean) / std, line)))
        return means, stds, np.array(processed_lines).T

    def revert(self, value):
        return value * self.stds[7:] + self.means[7:]
