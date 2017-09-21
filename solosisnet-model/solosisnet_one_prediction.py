import pandas as pd
import numpy as np
from keras.models import load_model

# Main Program
if __name__ == '__main__':

    # Load model...
    solosisnet = load_model("solosisnet.h5")

    stats = pd.read_csv('data/formated_stats.csv', index_col='#')

    # Set combat to predict:
    pok_A_id, pok_B_id = 100, 345

    # Format net input
    stats_A = stats[stats.index == pok_A_id].values[0].tolist()
    stats_B = stats[stats.index == pok_B_id].values[0].tolist()
    net_input = np.array([stats_A + stats_B])

    # Make a prediction...
    prediction = solosisnet.predict(net_input)

    if prediction > 0.5:
        # Send pokemon_A
        print('A')
    else:
        # Send pokemon_B
        print('B')
