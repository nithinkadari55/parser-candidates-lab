"""Stand-in for PubTabNet's src/parallel.py: metric.py imports it, but only batch_evaluate uses it (not used here)."""


def parallel_process(array, function, n_jobs=1, use_kwargs=False, front_num=0):
    return [function(**a) if use_kwargs else function(a) for a in array]
