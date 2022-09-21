import os
os.environ['LOGURU_LEVEL'] = 'ERROR'
import warnings

from lrtree import Lrtree

X, y, theta, BIC_oracle = Lrtree.generate_data(10000, 3, seed=1)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

model = Lrtree(class_num=4, max_iter=100)


if __name__ == "__main__":
    model.fit(X=X, y=y, nb_init=1, tree_depth=2, solver='newton-cg')
