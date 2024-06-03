import datetime
import numpy as np

def handle_non_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        raise TypeError("Unserializable object {} of type {}".format(obj, type(obj)))

def get_today_date_str():
    # Récupérer la date du jour
    today = datetime.datetime.now()
    # Formater la date en 'AAMMDD'
    return today.strftime('%y%m%d')