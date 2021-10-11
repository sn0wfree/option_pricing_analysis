# coding=utf-8
def timer(func):
    from functools import wraps
    import time

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        log_print_func = globals().get('Logger', None)
        if log_print_func is not None:
            print_func = log_print_func
        else:
            print_func = print
        end = time.time()
        print_func(func.__name__ + ' spend: ' + str(end - start))
        return res

    return wrapper


if __name__ == '__main__':
    @timer
    def test1(a):
        return a + 1


    test1(1)

    pass
