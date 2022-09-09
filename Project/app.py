from asyncio.windows_utils import pipe
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib
pipe = joblib.load(open('emotion_classifier_pipe_09_sept_2022.pkl',"rb"))


def predict_emotion(docx):
    result = pipe.predict([docx])
    return result[0]

def get_prediction_proba(docx):
    result = pipe.predict_proba([docx])
    return result   

emotion_emoji_dict = {"anger":"😡","disgust":"🤮",'fear':"😱","joy":"😃","neutral":"😑",'sadness':"🥺☹️", 'shame':"😣",'surprise':"😲🎊"}    

def main():
    st.title = "Emotion Classifier App"
    menu = ['Home','Moniter','About']
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == 'Home':
        st.subheader("Emotion In Text")

        with st.form(key = 'emotion_clf_form'):
            raw_text = st.text_area('Type Here')
            submit_text = st.form_submit_button(label='Submit')

        if submit_text:
            col1,col2 = st.columns(2)

            Prediction = predict_emotion(raw_text)
            Probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")  
                st.write(raw_text)
                st.success("Prediction")  
                emoji_icon = emotion_emoji_dict[Prediction]
                st.write("{0}:{1}".format(Prediction,emoji_icon))
                st.write("Confidence:{}".format(np.max(Probability)))

            with col2:
                st.success("Prediction Probability") 
                st.write(Probability)
                proba_df = pd.DataFrame(Probability,columns=pipe.classes_)
                st.write(proba_df.T)   
                proba_df_clean = proba_df.T.reset_index()
                proba_df_clean.columns = ["Emotion","Probability"]

                fig = alt.Chart(proba_df_clean).mark_bar().encode(x='Emotion',y='Probability',color = 'Emotion')
                st.altair_chart(fig,use_container_width = True)

    elif choice == 'Moniter':
        st.subheader("Moniter App")
    else:
        st.subheader("About")

if __name__ == '__main__':
    main()                