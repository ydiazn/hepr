from almiky.attacks import noise

from src import factories
from src.optimization import functions


hider_factories = {
    "qk": factories.QKrawtchoukBitHiderFactory,
    "dct": factories.DCTBitHiderFactory,
    "qct": factories.QCharlierThebichefBitHiderFactory
}

attack_callables = {
    "salt_pepper": noise.salt_paper_noise
}

objective_functions = {
    "cwa": functions.weighted_agregation,
    "dwa": functions.dynamic_weighted_agregation
}


def get_counter():
    '''
    Return a counter

    c = get_counter()
    a() => 0
    a() => 1
    ...
    '''
    iteration = 0

    def wrapper():
        nonlocal iteration
        iteration += 1
        return iteration

    return wrapper


def get_attack(attack, *args, **kwargs):
    def wrapper(ws_work):
        return attack(ws_work, *args, **kwargs)

    return wrapper
