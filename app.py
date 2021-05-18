import streamlit as st
import pandas as pd
import pickle 
from sklearn.preprocessing import StandardScaler as Scaler
from PIL import Image

credit_df = pd.read_csv('creditcard.csv')

st.markdown('# __Credit Card Fraud Detection__')
st.markdown('---')

#Sidebar
st.sidebar.markdown('## **User Input Features**')
st.sidebar.markdown('---')

def user_input_feature():
    X = {}
    X['V1'] = st.sidebar.slider('V1', credit_df['V1'].min(), credit_df['V1'].max(), credit_df['V1'].mean())
    X['V2'] = st.sidebar.slider('V2', credit_df['V2'].min(), credit_df['V2'].max(), credit_df['V2'].mean())
    X['V3'] = st.sidebar.slider('V3', credit_df['V3'].min(), credit_df['V3'].max(), credit_df['V3'].mean())
    X['V4'] = st.sidebar.slider('V4', credit_df['V4'].min(), credit_df['V4'].max(), credit_df['V4'].mean())
    X['V5'] = st.sidebar.slider('V5', credit_df['V5'].min(), credit_df['V5'].max(), credit_df['V5'].mean())
    X['V6'] = st.sidebar.slider('V6', credit_df['V6'].min(), credit_df['V6'].max(), credit_df['V6'].mean())
    X['V7'] = st.sidebar.slider('V7', credit_df['V7'].min(), credit_df['V7'].max(), credit_df['V7'].mean())
    X['V8'] = st.sidebar.slider('V8', credit_df['V8'].min(), credit_df['V8'].max(), credit_df['V8'].mean())
    X['V9'] = st.sidebar.slider('V9', credit_df['V9'].min(), credit_df['V9'].max(), credit_df['V9'].mean())
    X['V10'] = st.sidebar.slider('V10', credit_df['V10'].min(), credit_df['V10'].max(), credit_df['V10'].mean())
    X['V11'] = st.sidebar.slider('V11', credit_df['V11'].min(), credit_df['V11'].max(), credit_df['V11'].mean())
    X['V12'] = st.sidebar.slider('V12', credit_df['V12'].min(), credit_df['V12'].max(), credit_df['V12'].mean())
    X['V13'] = st.sidebar.slider('V13', credit_df['V13'].min(), credit_df['V13'].max(), credit_df['V13'].mean())
    X['V14'] = st.sidebar.slider('V14', credit_df['V14'].min(), credit_df['V14'].max(), credit_df['V14'].mean())
    X['V15'] = st.sidebar.slider('V15', credit_df['V15'].min(), credit_df['V15'].max(), credit_df['V15'].mean())
    X['V16'] = st.sidebar.slider('V16', credit_df['V16'].min(), credit_df['V16'].max(), credit_df['V16'].mean())
    X['V17'] = st.sidebar.slider('V17', credit_df['V17'].min(), credit_df['V17'].max(), credit_df['V17'].mean())
    X['V18'] = st.sidebar.slider('V18', credit_df['V18'].min(), credit_df['V18'].max(), credit_df['V18'].mean())
    X['V19'] = st.sidebar.slider('V19', credit_df['V19'].min(), credit_df['V19'].max(), credit_df['V19'].mean())
    X['V20'] = st.sidebar.slider('V20', credit_df['V20'].min(), credit_df['V20'].max(), credit_df['V20'].mean())
    X['V21'] = st.sidebar.slider('V21', credit_df['V21'].min(), credit_df['V21'].max(), credit_df['V21'].mean())
    X['V22'] = st.sidebar.slider('V22', credit_df['V22'].min(), credit_df['V22'].max(), credit_df['V22'].mean())
    X['V23'] = st.sidebar.slider('V23', credit_df['V23'].min(), credit_df['V23'].max(), credit_df['V23'].mean())
    X['V24'] = st.sidebar.slider('V24', credit_df['V24'].min(), credit_df['V24'].max(), credit_df['V24'].mean())
    X['V25'] = st.sidebar.slider('V25', credit_df['V25'].min(), credit_df['V25'].max(), credit_df['V25'].mean())
    X['V26'] = st.sidebar.slider('V26', credit_df['V26'].min(), credit_df['V26'].max(), credit_df['V26'].mean())
    X['V27'] = st.sidebar.slider('V27', credit_df['V27'].min(), credit_df['V27'].max(), credit_df['V27'].mean())
    X['V28'] = st.sidebar.slider('V28', credit_df['V28'].min(), credit_df['V28'].max(), credit_df['V28'].mean())
    X['Amount'] = st.sidebar.slider('Amount', credit_df['Amount'].min(), credit_df['Amount'].max(), credit_df['Amount'].mean())
    # Convert X to DataFrame
    X = pd.DataFrame(X, index=[0])
    return X

df = user_input_feature()

# Select model
selected_model = st.selectbox('Model: ',
                    ['Logistic Regression', 'Linear Discriminant Analysis', 'K Nearest Neighbors',
                     'Decision Tree Classifier', 'Gaussian Naive Bayes', 'Support Vector Machine'])

if selected_model == 'Logistic Regression':
    model = pickle.load(open('./Models/credit_lr.pkl', 'rb'))
elif selected_model == 'Linear Discriminant Analysis':
    model = pickle.load(open('./Models/credit_lda.pkl', 'rb'))
elif selected_model == 'K Nearest Neighbors':
    model = pickle.load(open('./Models/credit_knn.pkl', 'rb'))
elif selected_model == 'Decision Tree Classifier':
    model = pickle.load(open('./Models/credit_clf.pkl', 'rb'))
elif selected_model == 'Gaussian Naive Bayes':
    model = pickle.load(open('./Models/credit_nb.pkl', 'rb'))
elif selected_model == 'Support Vector Machine':
    model = pickle.load(open('./Models/credit_svm.pkl', 'rb'))
            


#Normalize data before making prediction
sc = Scaler()
amount = df['Amount'].values
df['Amount'] = sc.fit_transform(amount.reshape(-1,1))

#Apply model to make prediction
def make_prediction():
    if st.button("Run Model"):
        pred = model.predict(df)
        class_names = ['Non-Fraud', 'Fraud']
     
        st.markdown(f'## Result: {class_names[pred[0]]}')
        
make_prediction()

# Show model's evaluation
def show_evaluation():
    if st.checkbox("Show Model's Evaluation"):
        if selected_model == 'Logistic Regression':
            st.write('**Logistic Regression** is a Machine Learning classification algorithm that is used to predict the probability of a categorical dependent variable. In logistic regression, the dependent variable is a binary variable that contains data coded as 1 or 0. In other words, the logistic regression model predicts P(Y=1) as a function of X.')
            st.write('**The model is evaluated on a test set randomly selected from 20% of the entire dataset.*')
            st.write('**Recall Score:** 0.55')
            image = Image.open('./Images/LR.png')
            st.image(image, use_column_width=False)
        elif selected_model == 'Linear Discriminant Analysis':
            st.write('**Linear Discriminant Analysis** is a dimensionality reduction technique which is commonly used for the supervised classification problems. It is used for modeling differences in groups i.e. separating two or more classes. It is used to project the features in higher dimension space into a lower dimension space.') 
            st.write('**The model is evaluated on a test set randomly selected from 20% of the entire dataset.*')
            st.write('**Recall Score:** 0.73')
            image = Image.open('./Images/LDA.png')
            st.image(image, use_column_width=False)
        elif selected_model == 'K Nearest Neighbors':
            st.write('**K-Nearest Neighbors** is a non-parametric method used for classification and regression. In K-NN classification, the output is a class membership. This algorithm relies on distance and an object is classified by a plurality vote of its neighbors (the least distances), with the object being assigned to the class most common among its k nearest neighbors.')
            st.write('**The model is evaluated on a test set randomly selected from 20% of the entire dataset.*')
            st.write('**Recall Score:** 0.78')
            image = Image.open('./Images/KNN.png')
            st.image(image, use_column_width=False)
        elif selected_model == 'Decision Tree Classifier':
            st.write('**Decision Tree Classifier** is a type of Supervised Machine Learning method where the data is continuously split according to a certain parameter. The tree can be explained by two entities, namely decision nodes and leaves. The leaves are the decisions or the final outcomes. And the decision nodes are where the data is split.')
            st.write('**The model is evaluated on a test set randomly selected from 20% of the entire dataset.*')
            st.write('**Recall Score:** 0.71')
            image = Image.open('./Images/CART.png')
            st.image(image, use_column_width=False)
        elif selected_model == 'Gaussian Naive Bayes':
            st.write('**Gaussian Naive Bayes** is a Machine Learning algorithm in which continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution. A Gaussian distribution is also called Normal distribution.')
            st.write('**The model is evaluated on a test set randomly selected from 20% of the entire dataset.*')
            st.write('**Recall Score:** 0.75')
            image = Image.open('./Images/NB.png')
            st.image(image, use_column_width=False)
        elif selected_model == 'Support Vector Machine':
            st.write('**Support Vector Machine (SVM)** constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection. Intuitively, a good separation is achieved by the hyperplane that has the largest distance to the nearest training-data point of any class (functional margin).')
            st.write('**The model is evaluated on a test set randomly selected from 20% of the entire dataset.*')
            st.write('**Recall Score:** 0.64')
            image = Image.open('./Images/SVM.png')
            st.image(image, use_column_width=False)    

show_evaluation()
    