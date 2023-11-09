import os
import inspect
from ageml.datasets.synthetic_test_data import generate_synthetic_data as gsd
if __name__ == "__main__":
    # Set path in datasets
    p = os.path.join(os.path.dirname(inspect.getfile(gsd)),
                     "synthetic_dataset.csv")
    gsd(p)