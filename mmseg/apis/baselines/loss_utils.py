def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)