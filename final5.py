# import ssl
# import nltk

# # Create an unverified SSL context
# ssl._create_default_https_context = ssl._create_unverified_context

# # Download the 'stopwords'
# nltk.download('stopwords')
# nltk.download('punkt')

# import pandas as pd
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load the resume dataset
# df_resumes = pd.read_csv('/content/cleaned_resumes.csv')

# # Display a sample of the dataset
# print("Resume Data Sample:")
# print(df_resumes.head())


# # Define stop words and lemmatizer
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# # Preprocessing function
# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ''

#     # Basic cleaning and tokenization
#     text = re.sub(r'\W', ' ', text)  # Remove non-alphabet characters
#     text = re.sub(r'\d+', '', text)  # Remove digits
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     text = text.lower()  # Lowercase the text
#     tokens = word_tokenize(text)  # Tokenize the text
#     # Lemmatize and remove stop words
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
#     return ' '.join(tokens)

# # Apply preprocessing to the resume text column
# df_resumes['Cleaned_Text'] = df_resumes['Cleaned_Text'].apply(preprocess_text)

# # Display cleaned resumes
# print("Resumes After Preprocessing:")
# print(df_resumes['Cleaned_Text'].head())




# # Load the pre-trained SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Encode the resumes into vectors
# resume_vectors = model.encode(df_resumes['Cleaned_Text'].dropna().tolist())

# # Display shape of the encoded resume vectors
# print(f"Shape of resume vectors: {resume_vectors.shape}")



# # Load the pre-trained SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Encode the resumes into vectors
# resume_vectors = model.encode(df_resumes['Cleaned_Text'].dropna().tolist())

# # Display shape of the encoded resume vectors
# print(f"Shape of resume vectors: {resume_vectors.shape}")




# # Input the job description from the recruiter
# job_description = input("Please enter the job description: ")

# # Preprocess the job description
# cleaned_job_description = preprocess_text(job_description)

# # Encode the job description using the same model
# job_vector = model.encode([cleaned_job_description])

# # Display the cleaned and encoded job description
# print(f"Cleaned Job Description: {cleaned_job_description}")


# # Compute the cosine similarity between the job description and resume vectors
# similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]

# # Display the top similarity scores
# print(f"Similarity scores: {similarity_scores}")


# # Get the indices of the top 5 resumes with highest similarity
# top_n = 5
# top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

# # Retrieve the top 5 resumes
# top_resumes = df_resumes.iloc[top_indices]

# # Display the top 5 resumes and their similarity scores
# for i, (index, row) in enumerate(top_resumes.iterrows()):
#     print(f"\nResume {i+1}:")
#     print(f"Text: {row['Cleaned_Text']}")
#     print(f"Similarity Score: {similarity_scores[index]:.4f}")


# # Check the columns in the resumes dataframe
# print(df_resumes.columns)


# # Assuming df_resumes has a column 'File_Name' that holds the filenames of the resumes
# df_resumes['Filename'] = df_resumes['Filename'].fillna('Unnamed')  # Fallback for missing file names

# # Get the indices of the top 5 resumes with highest similarity
# top_n = 5
# top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

# # Retrieve the top 5 resumes with their file names
# top_resumes = df_resumes.iloc[top_indices]

# # Display the top 5 resumes, their file names, and similarity scores
# for i, (index, row) in enumerate(top_resumes.iterrows()):
#     print(f"\nResume {i+1}:")
#     print(f"Filename: {row['Filename']}")
#     print(f"Resume Text: {row['Cleaned_Text']}")
#     print(f"Similarity Score: {similarity_scores[index]:.4f}")









# import streamlit as st
# import pandas as pd
# import numpy as np
# import re
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# import ssl
# import nltk

# # Create an unverified SSL context
# ssl._create_default_https_context = ssl._create_unverified_context
# # Download necessary NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load the pre-trained SentenceTransformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # Define stop words and lemmatizer
# stop_words = set(stopwords.words('english'))
# lemmatizer = WordNetLemmatizer()

# # Preprocessing function
# def preprocess_text(text):
#     if not isinstance(text, str):
#         return ''
#     text = re.sub(r'\W', ' ', text)  # Remove non-alphabet characters
#     text = re.sub(r'\d+', '', text)  # Remove digits
#     text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
#     text = text.lower()  # Lowercase the text
#     tokens = word_tokenize(text)
#     tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
#     return ' '.join(tokens)

# # Load the resume dataset
# @st.cache_data
# def load_data():
#     df_resumes = pd.read_csv('cleaned_resumes.csv')  # Replace with your file path
#     df_resumes['Cleaned_Text'] = df_resumes['Cleaned_Text'].apply(preprocess_text)
#     return df_resumes

# # Main function to run the app
# def main():
#     st.title("Resume Matcher")
#     st.write("Match job descriptions with the most relevant resumes.")

#     # Load data
#     df_resumes = load_data()

#     # Job description input
#     job_description = st.text_area("Enter the job description", height=200)

#     if st.button("Find Matching Resumes"):
#         if job_description:
#             # Preprocess the job description
#             cleaned_job_description = preprocess_text(job_description)

#             # Encode the job description using the same model
#             job_vector = model.encode([cleaned_job_description])

#             # Encode the resumes into vectors
#             resume_vectors = model.encode(df_resumes['Cleaned_Text'].dropna().tolist())

#             # Compute the cosine similarity between the job description and resume vectors
#             similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]

#             # Get the indices of the top 5 resumes with highest similarity
#             top_n = 5
#             top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

#             # Retrieve the top 5 resumes
#             top_resumes = df_resumes.iloc[top_indices]

#             # Display the results
#             st.subheader("Top Matching Resumes")
#             for i, (index, row) in enumerate(top_resumes.iterrows()):
#                 st.write(f"**Resume {i+1}:**")
#                 st.write(f"- Filename: {row['Filename']}")
#                 st.write(f"- Similarity Score: {similarity_scores[index]:.4f}")
#                 st.write(f"- Text: {row['Cleaned_Text'][:500]}...")  # Displaying first 500 characters of the resume text
#         else:
#             st.warning("Please enter a job description to find matching resumes.")

# # Run the app
# if __name__ == '__main__':
#     main()




import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained SentenceTransformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Load the resume dataset
@st.cache_data
def load_data():
    # Define the URL for the CSV file
    csv_url = 'https://resumedataset.s3.eu-north-1.amazonaws.com/cleaned_resumes.csv'

    # Send a GET request to download the CSV file
    response = requests.get(csv_url)
    
    # Check if the request was successful
    if response.status_code != 200:
        st.error(f"Failed to download the dataset. Status code: {response.status_code}")
        return pd.DataFrame()  # Return an empty DataFrame if the request fails

    # Use io.StringIO to read the content directly from the downloaded file
    df_resumes = pd.read_csv(io.StringIO(response.text))
    df_resumes['Cleaned_Text'] = df_resumes['Cleaned_Text'].fillna('')  # Fill NaN with empty strings
    
    return df_resumes

# Main function to run the app
def main():
    st.title("Resume Matcher")
    st.write("Match job descriptions with the most relevant resumes.")

    # Load data and model
    df_resumes = load_data()
    model = load_model()

    # Job description input
    job_description = st.text_area("Enter the job description", height=200)

    if st.button("Find Matching Resumes"):
        if job_description:
            with st.spinner('Finding matching resumes, please wait...'):
                # Encode the job description using the model
                job_vector = model.encode([job_description])

                # Encode the resumes into vectors
                resume_vectors = model.encode(df_resumes['Cleaned_Text'].tolist())

                # Compute the cosine similarity between the job description and resume vectors
                similarity_scores = cosine_similarity(job_vector, resume_vectors)[0]

                # Get the indices of the top 5 resumes with the highest similarity
                top_n = 5
                top_indices = np.argsort(similarity_scores)[-top_n:][::-1]

                # Retrieve the top 5 resumes
                top_resumes = df_resumes.iloc[top_indices]

                # Display the results
                st.subheader("Top Matching Resumes")
                for i, (index, row) in enumerate(top_resumes.iterrows()):
                    st.write(f"*Resume {i+1}:*")
                    st.write(f"- Filename: {row['Filename']}")
                    st.write(f"- Similarity Score: {similarity_scores[index]:.4f}")
                    st.write(f"- Text: {row['Cleaned_Text'][:500]}...")  # Displaying first 500 characters of the resume text

        else:
            st.warning("Please enter a job description to find matching resumes.")

# Run the app
if __name__ == '__main__':
    main()
