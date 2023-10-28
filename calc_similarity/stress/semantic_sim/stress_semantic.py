import gensim
from gensim.models import word2vec
import numpy as np
import pandas as pd


a = ['Sulphur deficiency', 'Abscisic acid', 'Cold', 'Drought', 'Salt',
        'Compression', 'Tension', 'Alkalinity', 'Phosphate deficiency',
        'Cadmium', 'Heat', 'Copper', 'Waterlogging', 'Arsenic',
        'Aluminum excess', 'Hypoxic stress', 'Glyphosate',
        'Ferrum deficiency', 'Manganese', 'Cropping', 'Auxin',
        'Nitrogen deficiency', 'Topping', 'Wounding', 'Sucrose', 'Ozone',
        'Gibberellin', 'Osmotic stress', 'Heavy metal', 'Dehydration',
        'Arsenic and Selenium', 'Selenium', 'Lipopolysaccharide',
        'Calcium starvation', 'Potassium deficiency', 'sodium starvation',
        'Jasmonic Acid', 'Chromium', 'sugar accumulation',
        'Oxidative stress', 'Blue light', 'sulfur dioxide', 'Alkalinity Salt',
        'Gamma irradiated', 'Osmotic Stress', 'Drought then reWatering',
        'Darkness', 'Methyl jasmonate', 'Salicylic acid',
        'Ultraviolet radiation B', 'High light', 'Sound',
        'Low Energy Nitrogen Ion Irradiation', 'Nickel', 'Brassinosteroid']

sentences = [[word] for word in a]
#print(sentences)

model = gensim.models.Word2Vec(sentences, min_count=1,seed=42)

#print(model.wv['Nickel'])

c = []
for i in a:
    for j in a:
         c.append(model.wv.similarity(i, j))

#np.savetxt(r".\calc_similarity\stress\stress_sem_sim.txt",c)
