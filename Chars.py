import os
from sklearn.externals import joblib


def prediction(npaROIResized):
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_dir = os.path.join(current_dir, 'models/svc/svc.pkl')
    model = joblib.load(model_dir)
    image1d = npaROIResized.reshape(1, -1)
    result = model.predict(image1d)
    return result
