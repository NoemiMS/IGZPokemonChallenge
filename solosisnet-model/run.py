#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from keras.models import load_model
from flask import request, url_for
from flask_api import FlaskAPI, status, exceptions

# Start FlaskAPI
app = FlaskAPI(__name__)

@app.route("/", methods=['GET','POST'])
def pokemons_combat():
    """
    Create pokemons combat.
    """
    if request.method == 'POST':

        # Load model...
        solosisnet = load_model("solosisnet.h5")

        stats = pd.read_csv('data/formated_stats.csv', index_col='#')

        # Set combat to predict:
        pok_A_id = request.data.get('pokA', '')
        pok_B_id = request.data.get('pokB', '')

        # Format net input
        stats_A = stats[stats.index == pok_A_id].values[0].tolist()
        stats_B = stats[stats.index == pok_B_id].values[0].tolist()
        net_input = np.array([stats_A + stats_B])

        # Make a prediction...
        prediction = solosisnet.predict(net_input)

        if prediction < 0.5:
            # Send pokemon_A
            return [{'winner': 'A'}], status.HTTP_200_OK
        else:
            # Send pokemon_B
            return [{'winner': 'B'}], status.HTTP_200_OK

    # request.method == 'NOT_SUPPORTED'
    return [status.HTTP_415_UNSUPPORTED_MEDIA_TYPE]

if __name__ == '__main__':    
    # Debug true
    app.run(debug=True)
