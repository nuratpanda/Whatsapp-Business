import streamlit as st
import numpy as np
import string 
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import ast
st.set_option('deprecation.showfileUploaderEncoding',False)


whatsapp = pd.read_csv('whatsweb.csv')

def main():
    image = Image.open('WBL-logos_transparent.png')
    st.image(image)
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 200px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 200px;
        margin-left: -200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.markdown("<h1 style='text-align: center; color: Black;background-color:#e6e6fa'>Whatsapp Business Locator</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: White;'>Get products and services at the touch of a button😊</h3>", unsafe_allow_html=True)
    st.header("About")
    st.markdown("<h3 style='text-align: center; color: White;>An App thats brings business and services closer to you!!!</p>",unsafe_allow_html=True)
    # st.text(about)

   

    def remove_whitespace(business_type):
        if len(business_type) > 0:
            business_type = str(business_type)
            business_type = business_type.strip()
            return business_type
        else:
            st.text('Please enter a business type that you want recomendations based on')



    vectorizer = TfidfVectorizer(stop_words='english')
    word_vectors = vectorizer.fit_transform(whatsapp['Business_Type'])
    similarity_scores = cosine_similarity(word_vectors)
    indicies= pd.Series(whatsapp.index, index = whatsapp['Business_Type'])
    indicies = indicies.groupby(indicies.index).first()


    def recommend(m_des, similarity_scores, k = 5):
        m_id = indicies[m_des]
        similar_item_scores = list(enumerate(similarity_scores[m_id]))
        sorted_sim_scores = sorted(similar_item_scores, key = lambda x:x[1], reverse = True)[1:k+1]
        m_index = [idx[0] for idx in sorted_sim_scores]
        r_whatsapp = whatsapp.iloc[m_index]['Business_Type']
        r_numbers = whatsapp.iloc[m_index]['Number']
        r_address = whatsapp.iloc[m_index]['Business_Address']
        Biz_list = [r_whatsapp[x] for x in m_index]
        Num_list = [r_numbers[x] for x in m_index]
        add_list = [r_address[x] for x in m_index]
        final_series = pd.Series(index = [Num_list, add_list], data = Biz_list)
        final_dataframe = pd.DataFrame({'Number' : Num_list, 'Business Type' : Biz_list, 'Business_Address':add_list})
        return st.dataframe(final_dataframe)



    # Business = st.sidebar.text_input("What Products or Services do you need today?")
    
    Business_1 = st.sidebar.selectbox('Select',whatsapp['Business_Type'].values)
    Number = st.sidebar.slider("How many recommendations would you like?", 1, 10)
    Business = remove_whitespace(Business_1)
    if st.sidebar.button('Submit'):
        st.header(f'Recommendations for {Business} are :\n')
        recommend(Business, similarity_scores, Number)
if __name__ =='__main__':
    main() 

