from fastai.vision.all import *
from fastai.vision.widgets import *

path = Path()
learn_inf = load_learner(path/'export.pkl', cpu=True)

def predict(pic):
    label, _, probs = learn_inf.predict(pic)
    return label, probs.max().item()