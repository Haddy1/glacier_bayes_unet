import numpy as np
from predict import get_cutoff_point
from keras.models import load_model
from layers.BayesDropout import  BayesDropout
from pathlib import Path

path = Path('output_bayes')

for d in path:
    model_file = next(d.glob('model_*.h5'))

    model = load_model(str(model_file.absolute()), custom_objects={'BayesDropout':BayesDropout})
    if 'flip' in d.name:
        get_cutoff_point(model, 'data_256_flip/val', d, batch_size=4)
    else:
        get_cutoff_point(model, 'data_256/val', d, batch_size=4)



