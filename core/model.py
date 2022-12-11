import tensorflow as tf
import numpy as np

from .utility import load_qf_map, estimation_by_mse
from . import max_min_coefficient,\
    custom_softmax_activation,\
    custom_two_terms_loss_wrapper,\
    custom_mse_wrapper,\
    label2coefficient 


class Model:
    def __init__(self, model_path: str, qf1_qf2_map_path: str):
        self.qf_labels, self.qf_coeffs = load_qf_map(qf1_qf2_map_path)
        self.max_coeffs, self.min_coeffs = max_min_coefficient(quality_range=(50, 100),
                                                               n_coeffs=15,
                                                               zig_zag_order=True)
        self._init_model(model_path)

    def _init_model(self, model_path: str):
        custom_objects = {}
        custom_objects['custom_softmax'] = custom_softmax_activation(
            self.max_coeffs)
        custom_objects['custom_two_terms_loss_wrapper'] = custom_two_terms_loss_wrapper(
            self.max_coeffs, 0.8)
        custom_objects['custom_mse'] = custom_mse_wrapper(self.max_coeffs)

        self.model = tf.keras.models.load_model(
            model_path, custom_objects=custom_objects)
        
    def predict_coeffs(self, image):
        prediction = self.model.predict(np.expand_dims(image, [0, -1]))

        predicted_label = label2coefficient(
            prediction.flatten(), max_coefficients=self.max_coeffs)
        return predicted_label
        

    def estimate_qf1(self, image, estimator: callable = estimation_by_mse):
        predicted_label = self.predict_coeffs(image)
        argmin = np.argmin(estimator(predicted_label, self.qf_coeffs))
        return self.qf_labels[argmin]

