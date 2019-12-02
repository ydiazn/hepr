from src.nets import regression

def get_net(name, **kwargs):
    if name == 'regression':
        clss = regression.RegressionNet
    elif name == 'regression2':
        clss = regression.RegressionNet2
    else:
        raise NotImplementedError('%s net was not supported' % name)

    return clss(**kwargs)