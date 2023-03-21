import numpy as np
import tensorflow as tf
from tensorflow import keras
from ctc import CTCLoss
from spectrogram import spectrogram_batch

MODEL_PATH = './Saved_Models/v1.1_17_4_7'


class _stt:
    """
    This is a singleton class i.e. it can only have one instance
    """
    model = None
    instance = None
    _mappings = [
        ' ', u'\u0900', u'\u0901', u'\u0902', u'\u0903', u'\u0904', u'\u0905', u'\u0906', u'\u0907', u'\u0908',
        u'\u0909', u'\u090A', u'\u090B', u'\u090C', u'\u090D', u'\u090E', u'\u090F',
        u'\u0910', u'\u0911', u'\u0912', u'\u0913', u'\u0914', u'\u0915', u'\u0916', u'\u0917', u'\u0918', u'\u0919',
        u'\u091A', u'\u091B', u'\u091C', u'\u091D', u'\u091E', u'\u091F',
        u'\u0920', u'\u0921', u'\u0922', u'\u0923', u'\u0924', u'\u0925', u'\u0926', u'\u0927', u'\u0928', u'\u0929',
        u'\u092A', u'\u092B', u'\u092C', u'\u092D', u'\u092E', u'\u092F',
        u'\u0930', u'\u0931', u'\u0932', u'\u0933', u'\u0934', u'\u0935', u'\u0936', u'\u0937', u'\u0938', u'\u0939',
        u'\u093A', u'\u093B', u'\u093C', u'\u093D', u'\u093E', u'\u093F',
        u'\u0940', u'\u0941', u'\u0942', u'\u0943', u'\u0944', u'\u0945', u'\u0946', u'\u0947', u'\u0948', u'\u0949',
        u'\u094A', u'\u094B', u'\u094C', u'\u094D', u'\u094E', u'\u094F',
        u'\u0950', u'\u0951', u'\u0952', u'\u0953', u'\u0954', u'\u0955', u'\u0956', u'\u0957', u'\u0958', u'\u0959',
        u'\u095A', u'\u095B', u'\u095C', u'\u095D', u'\u095E', u'\u095F',
        u'\u0960', u'\u0961', u'\u0962', u'\u0963', u'\u0964', u'\u0965', u'\u0966', u'\u0967', u'\u0968', u'\u0969',
        u'\u096A', u'\u096B', u'\u096C', u'\u096D', u'\u096E', u'\u096F',
        u'\u0970', u'\u0971', u'\u0972', u'\u0973', u'\u0974', u'\u0975', u'\u0976', u'\u0977', u'\u0978', u'\u0979',
        u'\u097A', u'\u097B', u'\u097C', u'\u097D', u'\u097E', u'\u097F'
    ]
    _char_to_num = keras.layers.StringLookup(vocabulary=_mappings, oov_token="", encoding='utf8')
    _num_to_char = keras.layers.StringLookup(
        vocabulary=_char_to_num.get_vocabulary(), oov_token="", invert=True, encoding='utf8'
    )

    def _decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output_text = []
        for result in results:
            result = tf.strings.reduce_join(self._num_to_char(result)).numpy().decode("utf-8")
            output_text.append(result)
        return output_text

    def predict(self, file_path):
        dataset = spectrogram_batch(file_path)
        predictions = []
        for batch in dataset:
            X = batch
            batch_predictions = self.model.predict(X)
            batch_predictions = self._decode_batch_predictions(batch_predictions)
            predictions.extend(batch_predictions)
        return predictions


def stt():
    # ensure that we only have one instance
    if _stt.instance is None:
        _stt.instance = _stt()
        _stt.model = keras.models.load_model(MODEL_PATH, custom_objects={"CTCLoss": CTCLoss})
    return _stt.instance
