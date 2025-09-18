# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 09:01:53 2025

@author: KETSAR
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from PIL import Image, ImageFile

st.set_page_config(
        page_title='Pinguin Classifier',
        page_icon='🐧'
    )

st.title('🐧 Pinguin Classifier App')
image = Image.open(r'penguin.jpg')
st.image(image)

st.info('Aplikasi ini menggunakan Machine Learning untuk memprediksi spesies pinguin berdasarkan ciri-cirinya.')

st.divider()

with st.expander('Informasi Awal'):
    st.write('Data Mentah')
    df = pd.read_csv('penguin.csv', index_col=0)
    df
    
    st.write('**Variabel Independen (X)**')
    X_data = df.drop('species', axis=1)
    X_data
    
    st.write('**Variabel Dependen (y)**')
    y_data = df.species
    y_data

with st.expander('Visualiasi Data'):
    st.scatter_chart(data=df, x='bill_length_mm', y='body_mass_g', color='species')
    
with st.sidebar:
    st.header('**Masukan Data**')
    island = st.selectbox('Pulau', ('Biscoe', 'Torgersen', 'Dream'))
    bill_length_mm = st.slider('Panjang Paruh (mm)', np.min(df.bill_length_mm), np.max(df.bill_length_mm), np.median(df.bill_length_mm))
    bill_depth_mm = st.slider('Kedalaman Paruh (mm)', np.min(df.bill_depth_mm), np.max(df.bill_depth_mm), np.median(df.bill_depth_mm))
    flipper_length_mm = st.slider('Panjang Sayap (mm)', float(np.min(df.flipper_length_mm)), float(np.max(df.flipper_length_mm)), float(np.median(df.flipper_length_mm)))
    body_mass_g = st.slider('Berat Tubuh (g)', float(np.min(df.body_mass_g)), float(np.max(df.body_mass_g)), float(np.median(df.body_mass_g)))
    sex = st.selectbox('Jenis Kelamin', ('male', 'female'))
    
    data = {'island' : island,
            'bill_length_mm' : bill_length_mm,
            'bill_depth_mm' : bill_depth_mm,
            'flipper_length_mm' : flipper_length_mm,
            'body_mass_g' : body_mass_g,
            'sex' : sex}
    input_df = pd.DataFrame(data, index=[0])
    input_penguins = pd.concat([input_df, X_data], axis=0).reset_index(drop=True)
    
with st.expander('Masukan Data User'):
    st.write('**Masukan Data Pengiun**')
    input_df
    st.write('**Kombinasi Data Pinguin**')
    input_penguins
    

# persiapan data
# encode X
encode = ['island', 'sex']
df_penguins = pd.get_dummies(input_penguins, prefix=encode)

X = df_penguins[1:]
input_row = df_penguins[:1]

target_mapper = {'Adelie': 0,
                 'Chinstrap': 1,
                 'Gentoo': 2}

def target_encode(val):
    return target_mapper[val]

y_encode = y_data.map(target_encode)
y_encode.name = 'y_encode'
y_data_named = y_data.copy()
y_data_named.name = 'y_decode'
df_y = pd.concat([y_encode, y_data_named], axis=1)

with st.expander('Persiapan Data'):
    st.write('**Encode Variabel X** ***(Input Penguin)***')
    input_row
    st.write('**Encode Variabel y** ***(Target)***')
    df_y
    
clf = RandomForestClassifier()
clf.fit(X, y_encode)    
    
y_pred = clf.predict(input_row)
y_pred_proba = clf.predict_proba(input_row)

df_pred_proba = pd.DataFrame(y_pred_proba)
df_pred_proba.columns = ['Adelie', 'Chinstrap', 'Gentoo']
df_pred_proba.rename({  0: 'Adelie',
                        1: 'Chinstrap',
                        2: 'Gentoo'})
st.divider()
# nampilin hasil prediksi spesies
st.subheader('Prediksi Spesies')
st.dataframe(df_pred_proba,
             column_config={
                 'Adelie': st.column_config.ProgressColumn(
                     'Adelie',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                     ),
                 'Chinstrap': st.column_config.ProgressColumn(
                     'Chinstrap',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                     ),
                 'Gentoo': st.column_config.ProgressColumn(
                     'Gentoo',
                     format='%f',
                     width='medium',
                     min_value=0,
                     max_value=1
                     )
                 }, hide_index=True)

penguin_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.success(str(penguin_species[y_pred[0]]))
    
    
    
    
    
    
    
    
    
