import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF
from scipy.spatial.distance import cosine
import plotly.express as px

def main():
    st.markdown("<h1 style='text-align: center;'>Cigarettes Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("#")

    # Display the data
    st.markdown("<h1 style='text-align: center; font-size: 30px;'>Cigarettes Data</h1>", unsafe_allow_html=True)
    st.write("")
    df = pd.read_csv('smokerdata.csv')
    st.dataframe(df)
    df['Brand_Variety'] = df['Brand'] + ' ' + df['Variety']
    df_content = df[df['Rating'] > 1].copy()
    df_content.head()
    st.write("")

    # Give the detail each of the data's columns
    st.write("""**:blue[Column descriptions]**\n
    User : User account name.
    Brand : Brand of a cigarette that user rates.
    Variety : Variety of a cigarette that user rates.
    Type : Type of a cigarette.
    Date : The date when user rates a cigarette.
    Strength : The strength of a cigarette.
    Taste : The taste of a cigarette.
    Price : The price of a cigarette.
    Rating : A score that user gives.""")
    st.write("#")

    # //=======================================================================================\\ #
    X = df_content[['Brand_Variety', 'Strength', 'Taste', 'Price']].copy()
    y = df_content['User'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    Brand_list = df_content['Brand_Variety'].unique().tolist()
    temp_list = []
    for i, brand in enumerate(Brand_list):
        temp_list.append([
            X_train[X_train['Brand_Variety'] == brand].mode()['Strength'][0],
            X_train[X_train['Brand_Variety'] == brand].mode()['Taste'][0],
            X_train[X_train['Brand_Variety'] == brand].mode()['Price'][0],
        ])
    cig_mode_df = pd.DataFrame(temp_list, columns=['Strength', 'Taste', 'Price'])
    cig_mode_df['Brand_Variety'] = Brand_list
    X = df[['Brand_Variety', 'Rating']].copy()
    y = df['User'].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)
    crosstab_matrix = pd.crosstab(y_train, X_train['Brand_Variety'], X_train['Rating'], aggfunc='mean')
    crosstab_matrix.fillna(value=0, inplace=True)
    crosstab_matrix.reset_index(drop=True, inplace=True)
    nmf = NMF(n_components=30)
    nmf.fit(crosstab_matrix)
    H = pd.DataFrame(np.round(nmf.components_, 2), columns=crosstab_matrix.columns)
    W = pd.DataFrame(np.round(nmf.transform(crosstab_matrix), 2), columns=H.index)
    recommend_matrix = pd.DataFrame(np.round(np.dot(W, H), 2), columns=H.columns)
    crosstab_matrix = pd.crosstab(y_train, X_train['Brand_Variety'], X_train['Rating'], aggfunc='mean')
    crosstab_matrix.fillna(value=0, inplace=True)
    crosstab_matrix.reset_index(drop=True, inplace=True)
    new_user_matrix = pd.crosstab(y_test, X_test['Brand_Variety'], X_test['Rating'], aggfunc='mean')
    new_user_matrix.fillna(value=0, inplace=True)
    new_user_matrix.reset_index(drop=True, inplace=True)
    diff_columns = set(crosstab_matrix.columns) - set(new_user_matrix.columns)
    for column in diff_columns:
        new_user_matrix = new_user_matrix.assign(column=[0]*new_user_matrix.shape[0])
    new_user_matrix = new_user_matrix.reindex(columns=crosstab_matrix.columns)
    new_user_matrix.fillna(value=0, inplace=True)
    # \\=======================================================================================// #

    # Model selection
    st.write("**:blue[Which recommendation method do you want to use?]**")
    model = st.selectbox(
        "none",
        ("Content-based filtering", "Collaborative filtering (model-based)", "Collaborative filtering (memory-based)"),
        label_visibility='collapsed')
    st.write("")
    if model == "Content-based filtering":
        STR = st.selectbox(
        '**:red[Strength]**',
        ('Very Strong','Strong', 'Medium',  'Weak', 'Very Weak')
        )
        Taste = st.selectbox(
            '**:violet[Taste]**',
            ('Very Pleasant', 'Pleasant', 'Tolerable', 'Poor', 'Very Poor')
        )
        Price = st.selectbox(
            '**:orange[Price]**',
            ('Very Low', 'Low', 'Fair', 'High', 'Very High')
        )
        topn = round(st.number_input('**Number of cigarettes**', min_value=1, step=1))
    # //=======================================================================================\\ #
        def cig_recommend(S, T, P, topn): # S stands for strength, T for taste, and P for price
            Str_list = ['Very Strong','Strong', 'Medium',  'Weak', 'Very Weak']
            Taste_list = ['Very Pleasant', 'Pleasant', 'Tolerable', 'Poor', 'Very Poor']
            Price_list = ['Very Low', 'Low', 'Fair', 'High', 'Very High']
            cig_df = cig_mode_df[(cig_mode_df['Strength'] == S) & (cig_mode_df['Taste'] == T) & (cig_mode_df['Price'] == P)].copy()
            if len(cig_df) < topn:
                for Str in Str_list:
                    if Str != S:
                        cig_df2 = cig_mode_df[(cig_mode_df['Strength'] == Str) & (cig_mode_df['Taste'] == T) & (cig_mode_df['Price'] == P)].copy()
                        cig_df = pd.concat([cig_df, cig_df2], ignore_index= True)
                    if len(cig_df) >= topn:
                        return cig_df[:topn]
                for Taste in Taste_list:
                    if Taste != T:
                        cig_df2 = cig_mode_df[(cig_mode_df['Strength'] == S) & (cig_mode_df['Taste'] == Taste) & (cig_mode_df['Price'] == P)].copy()
                        cig_df = pd.concat([cig_df, cig_df2], ignore_index= True)
                    if len(cig_df) >= topn:
                        return cig_df[:topn]
                for Price in Price_list:
                    if Price != P:
                        cig_df2 = cig_mode_df[(cig_mode_df['Strength'] == S) & (cig_mode_df['Taste'] == T) & (cig_mode_df['Price'] == Price)].copy()
                        cig_df = pd.concat([cig_df, cig_df2], ignore_index= True)
                    if len(cig_df) >= topn:
                        return cig_df[:topn]
            return cig_df[:topn]
    # \\=====================================================================================// #
        temp = cig_recommend(STR, Taste, Price, topn).reset_index(drop=True)
        if temp.shape[0] < topn:
            temp.index = range(1, temp.shape[0]+1)
        else:
            temp.index = range(1, topn+1)
        st.write("#")
        st.write("**:green[Here are cigarettes for you!]**")
        st.dataframe(temp)
    elif model == "Collaborative filtering (model-based)":
    # //======================================================================================\\ #
        def cig_recommend_model(uid, topn):
            rec_cig_brands = recommend_matrix.iloc[uid].sort_values(ascending=False)
            rec_cig_brands = rec_cig_brands[rec_cig_brands > 0].index
            recommend_df = pd.DataFrame(columns=cig_mode_df.columns)
            for brand in rec_cig_brands:
                recommend_df = pd.concat([recommend_df, cig_mode_df[cig_mode_df['Brand_Variety'] == brand]])
            return recommend_df[:topn]
    # \\=======================================================================================// #
        uid = round(st.number_input('**User id**',min_value=0, max_value=recommend_matrix.index[-1], step=1))
        # st.dataframe(pd.DataFrame(recommend_matrix.iloc[uid]).T)
        topn = round(st.number_input('**Number of cigarettes** ', min_value=1, step=1))
        temp = cig_recommend_model(uid, topn).reset_index(drop=True)
        temp.index = range(1, topn+1)
        st.write("#")
        st.write("**:green[Here are cigarettes for you!]**")
        st.dataframe(temp)
    else:
    # //=======================================================================================\\ #
        def cig_recommend_memory(uid, topn):
            similar_list = []
            target = new_user_matrix.iloc[uid]
            target_brands = target[target > 0].index
            for row in range(crosstab_matrix.shape[0]):
                similarity = 1 - cosine(target, crosstab_matrix.iloc[row]) # cosine([1,0,0], [0.5,0,0]) = 0
                if similarity != 1:
                    similar_list.append(similarity)
                else:
                    similar_list.append(0)
            most_similar_user = np.argsort(similar_list)[::-1]
            rec_cig_brands = [target_brands]
            for id in most_similar_user:
                temp = crosstab_matrix.iloc[id]
                brands = temp[temp > 0].index
                for brand in brands:
                    if len(rec_cig_brands) == topn+1:
                        recommend_df = pd.DataFrame(columns=cig_mode_df.columns)
                        for brand in rec_cig_brands[1:]:
                            recommend_df = pd.concat([recommend_df, cig_mode_df[cig_mode_df['Brand_Variety'] == brand]])
                        return recommend_df
                    if brand not in rec_cig_brands:
                        rec_cig_brands.append(brand)
            recommend_df = pd.DataFrame(columns=cig_mode_df.columns)
            for brand in rec_cig_brands:
                recommend_df = pd.concat([recommend_df, cig_mode_df[cig_mode_df['Brand_Variety'] == brand]])
            return recommend_df
    # \\=======================================================================================// #
        uid = round(st.number_input('**User id**',min_value=0, max_value=new_user_matrix.index[-1], step=1))
        # st.dataframe(pd.DataFrame(new_user_matrix.iloc[uid]).T)
        topn = round(st.number_input('**Number of cigarettes** ', min_value=1, step=1))
        st.write("#")
        st.write("**:green[Here are cigarettes for you!]**")
        temp = cig_recommend_memory(uid, topn)
        temp.index = range(1, topn+1)
        st.dataframe(temp)

if __name__ == "__main__":
    main()