import cProfile
import os
import pstats
import warnings

from lrtree import Lrtree

X, y, theta, BIC_oracle = Lrtree.generate_data(10000, 3, seed=1)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['LOGURU_LEVEL'] = 'ERROR'

model = Lrtree(class_num=4, max_iter=100)


if __name__ == "__main__":
    prof = cProfile.Profile()
    prof.enable()
    model.fit(X=X, y=y, nb_init=1, tree_depth=2)
    prof.disable()

    stats = pstats.Stats(prof).strip_dirs().sort_stats("cumtime")
    stats.print_stats(10)
