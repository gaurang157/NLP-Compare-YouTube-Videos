import streamlit as st
import youtube_transcript_api
import re
from collections import Counter
import re
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
# from multi_emotion import multi_emotion
from pysentimiento import create_analyzer
import pandas as pd
import json
from textblob import TextBlob
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock


# Function to get video transcript from YouTube

def get_youtube_transcript(video_url):
    try:
        video_id = video_url.split("v=")[1]
        video = youtube_transcript_api.YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([entry['text'] for entry in video])
        return transcript
    except Exception as e:
        return None

# Streamlit app
st.title("Compare ▶️ YouTube Videos")
st.write("Enter the URLs of YT ▶️ to process:")
'''e.g. YouTube Video Link ONLY'''
# st.sidebar.markdown("Process any Text🔡Web🕸️Page📄")
st.sidebar.link_button("NLP Web Content Analysis", "https://streamlit.io/gallery",use_container_width=True)

# st.sidebar.markdown("Compare any Text🔡Web🕸️Page📄")
st.sidebar.link_button("NLP Compare Web Content", "https://streamlit.io/gallery",use_container_width=True)

# st.sidebar.markdown("Process any YouTube ▶️ Video of English Language")
st.sidebar.link_button("NLP YouTube Analysis", "https://streamlit.io/gallery",use_container_width=True)

# st.sidebar.markdown("Compare any YT ▶️ with captions")
st.sidebar.link_button("NLP Compare YouTube Videos (Selected 🎉 ✅)", "https://streamlit.io/gallery",use_container_width=True)
col1, col2 = st.columns(2)
with col1:
    
    # Initialize with two URL input fields and checkboxes
    # user_urls = [st.text_input("URL 1")]
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    
    user_urls = [st.text_input(
        "URL 1",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder="e.g. https://www.youtube.com/watch?v=rwkPCt77Mcs",
        key="placeholder",
    )]
    
with col2:
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
    
    user_urls1 = [st.text_input(
        "URL 2",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder="e.g. https://www.youtube.com/watch?v=rwkPCt77Mcs",
        key="placeholder1",
    )]

if st.button("Process URLs"):
    st.write("Please wait a minute for amazing results. We'll start with Url1 and then proceed to Url2.")
    col11, col22 = st.columns(2)
    with col11:
        
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize, sent_tokenize
        nltk.download('stopwords')
        nltk.download('punkt')
        stopWords = set(stopwords.words("english"))
        # Function to summarize a given text
        def summarize_text(text):
            # Tokenizing the text
            words = word_tokenize(text)
        
            # Creating a frequency table to keep the score of each word
            freqTable = dict()
            for word in words:
                word = word.lower()
                if word in stopWords:
                    continue
                if word in freqTable:
                    freqTable[word] += 1
                else:
                    freqTable[word] = 1
        
            # Creating a dictionary to keep the score of each sentence
            sentences = sent_tokenize(text)
            sentenceValue = dict()
        
            for sentence in sentences:
                for word, freq in freqTable.items():
                    if word in sentence.lower():
                        if sentence in sentenceValue:
                            sentenceValue[sentence] += freq
                        else:
                            sentenceValue[sentence] = freq
        
            sumValues = 0
            for sentence in sentenceValue:
                sumValues += sentenceValue[sentence]
        
            # Average value of a sentence from the original text
            average = int(sumValues / len(sentenceValue))
        
            # Storing sentences into our summary.
            summary = ''
            for sentence in sentences:
                if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
                    summary += " " + sentence
        
            return summary

        # Function to extract common keywords from a given text
        def extract_keywords(text, top_n=10):
            # Preprocess and remove stopwords
            filtered_words = []
            words = nltk.word_tokenize(text)
            for word in words:
                word = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', word)
                if word.lower() not in stopWords and len(word) > 1:
                    filtered_words.append(word.lower())
        
            # Extract keywords (non-stopwords as keywords)
            keywords = [word for word in filtered_words]
        
            # Count word frequencies
            word_count = Counter(keywords)
        
            # Extract the top N keywords based on their frequency
            top_keywords = [keyword for keyword, _ in word_count.most_common(top_n)]
        
            return top_keywords
        
        user_video_urls = [url for url in user_urls if url.strip()]
        user_transcripts = []

        for url in user_video_urls:
            transcript = get_youtube_transcript(url)

            if transcript:
                user_transcripts.append(transcript)
        # Convert the list of transcripts into a single string
        text = "\n".join(user_transcripts)
        
        
        
        # from transformers import AddedToken
        
        # # Define a custom hash function for tokenizers.AddedToken
        # def my_hash_func(token):
        #     try:
        #         return hash((token.ids, token.type_id))
        #     except AttributeError:
        #         # Handle cases where the token object is not as expected
        #         return hash(str(token))
        
        # @st.cache_resource(hash_funcs={AddedToken: my_hash_func})
        # def get_analyzers():
        #     from setup import analyzer, emotion_analyzer, hate_speech_analyzer
        #     return analyzer, emotion_analyzer, hate_speech_analyzer
        from my_module import get_analyzers
        
        # Load analyzers
        analyzers = get_analyzers()
        
        # Now you can use the analyzers for text analysis
        sentiment1 = analyzers[0].predict(text)
        emotion1 = analyzers[1].predict(text)
        hate_speech1 = analyzers[2].predict(text)
        # from setup import analyzer, emotion_analyzer, hate_speech_analyzer
        # from transformers import AddedToken
        
        # # Define a custom hash function for tokenizers.AddedToken
        # def my_hash_func(token):
        #     try:
        #         return hash((token.ids, token.type_id))
        #     except AttributeError:
        #         # Handle cases where the token object is not as expected
        #         return hash(str(token))
        
        # @st.cache_data(hash_funcs={AddedToken: my_hash_func})
        # def create_analyzers():
        #     return analyzer, emotion_analyzer, hate_speech_analyzer
        
        # analyzers = create_analyzers()
        # sentiment1 = analyzers[0].predict(text)
        # emotion1 = analyzers[1].predict(text)
        # hate_speech1 = analyzers[2].predict(text)

        TOInews = re.sub("[^A-Za-z" "]+", " ", text).lower()
        TOInews_tokens = TOInews.split(" ")

        with open("en-stop-words.txt", "r") as sw:
            stop_words = sw.read()
                        
        stop_words = stop_words.split("\n")

        tokens = [w for w in TOInews_tokens if not w in stop_words]

        tokens_frequencies = Counter(tokens)

        # tokens_frequencies = tokens_frequencies.loc[tokens_frequencies.text != "", :]
        # tokens_frequencies = tokens_frequencies.iloc[1:]

        # Sorting
        tokens_frequencies = sorted(tokens_frequencies.items(), key = lambda x: x[1])

        # Storing frequencies and items in separate variables 
        frequencies = list(reversed([i[1] for i in tokens_frequencies]))
        words = list(reversed([i[0] for i in tokens_frequencies]))

        # Barplot of top 10 
        # import matplotlib.pyplot as plt
        
        
        # Create a figure and bar chart
        with _lock:
            plt.figure(figsize=(8, 4))
            plt.bar(height=frequencies[0:11], x=range(0, 11), color=['red', 'green', 'black', 'yellow', 'blue', 'pink', 'violet'], width=0.6)
            plt.title("Top 10 Tokens (Words)")
            plt.grid(True)
            # Customize the x-axis labels and rotation for visibility
            plt.xticks(range(0, 11), words[0:11], rotation=45)
            plt.xlabel("Tokens")
            plt.ylabel("Count")
            
            # Display the plot in Streamlit
            st.pyplot(plt, use_container_width=True)
        ##########

        st.write("Please be patience for Amazing Results it will take few minutes")
        # Joinining all the tokens into single paragraph 
        cleanstrng = " ".join(words)

        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_ip = WordCloud(background_color = 'White', width = 2800, height = 2400).generate(cleanstrng)
            plt.title("Normal Word Cloud")
            plt.axis("off")
            plt.grid(False)
            plt.imshow(wordcloud_ip)
            st.pyplot(plt, use_container_width=True)


        #########################################################################################

        # positive words
        with open("en-positive-words.txt", "r") as pos:
            poswords = pos.read().split("\n")
        # Positive word cloud
        # Choosing the only words which are present in positive words
        pos_tokens = " ".join ([w for w in TOInews_tokens if w in poswords])

        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_positive = WordCloud(background_color = 'White', width = 1800, height = 1400).generate(pos_tokens)
            plt.title("Positive Word Cloud")
            plt.axis("off")
            plt.grid(False)
            plt.imshow(wordcloud_positive)
            st.pyplot(plt, use_container_width=True)

        # Negative words
        with open("en-negative-words.txt", "r") as neg:
            negwords = neg.read().split("\n")
        # Negative word cloud
        # Choosing the only words which are present in negwords
        neg_tokens = " ".join ([w for w in TOInews_tokens if w in negwords])
        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_negative = WordCloud(background_color = 'black', width = 1800, height=1400).generate(neg_tokens)
            plt.title("Negative Word Cloud")
            plt.axis("off")
            plt.grid(False)
            plt.imshow(wordcloud_negative)
            st.pyplot(plt, use_container_width=True)
        #########################################################################################
        
        
        # Word cloud with 2 words together being repeated

        # Extracting n-grams using TextBlob

        bigrams_list = list(nltk.bigrams(tokens))
        dictionary2 = [' '.join(tup) for tup in bigrams_list]

        # Using count vectorizer to view the frequency of bigrams
        
        vectorizer = CountVectorizer(ngram_range = (2, 2))
        bag_of_words = vectorizer.fit_transform(dictionary2)
        v1 = vectorizer.vocabulary_

        sum_words = bag_of_words.sum(axis = 0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in v1.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

        words_dict = dict(words_freq[:100])
        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_2 = WordCloud(background_color = 'black', width = 1800, height = 1400)                 
            wordcloud_2.generate_from_frequencies(words_dict)
            plt.title("Bi-Gram based on Frequency")
            plt.axis("off")
            plt.grid(False)
            plt.imshow(wordcloud_2)
            st.pyplot(plt, use_container_width=True)
        ##############################################################################################
        
        # Word cloud with 2 words together being repeated
        
        # Extracting n-grams using TextBlob

        bigrams_list2 = list(nltk.trigrams(tokens))
        dictionary3 = [' '.join(tup) for tup in bigrams_list2]

        # Using count vectorizer to view the frequency of bigrams
        
        vectorizer1 = CountVectorizer(ngram_range = (3, 3))
        bag_of_words1 = vectorizer1.fit_transform(dictionary3)
        v2 = vectorizer1.vocabulary_

        sum_words1 = bag_of_words1.sum(axis = 0)
        words_freq1 = [(word1, sum_words1[0, idx1]) for word1, idx1 in v2.items()]
        words_freq1 = sorted(words_freq1, key = lambda x: x[1], reverse = True)

        words_dict1 = dict(words_freq1[:100])
        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_3 = WordCloud(background_color = 'black', width = 1800, height = 1400)                  
            wordcloud_3.generate_from_frequencies(words_dict1)
            plt.title("Tri-Gram based on Frequency")
            plt.grid(False)
            plt.axis("off")
            plt.imshow(wordcloud_3)
            st.pyplot(plt, use_container_width=True)

        # eqn shift 1
        pattern = "[^A-Za-z.]+"

        # Perform text preprocessing without removing full stops
        sen = re.sub(pattern, " ", text).lower()

        # SENTANCE Tokenizer
        sen_t = sen.split(".")


        # Create a DataFrame with the sentences as lists
        df = pd.DataFrame(sen_t)

        # Display the DataFrame
        print(df)

        df.columns = ['text']
        

        # Number of words
        df['number_of_words'] = df['text'].apply(lambda x : len(TextBlob(x).words))

        # Detect presence of wh words
        wh_words = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])
        df['are_wh_words_present'] = df['text'].apply(lambda x : True if len(set(TextBlob(str(x)).words).intersection(wh_words)) > 0 else False)


        # Polarity
        df['polarity'] = df['text'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)

        # Subjectivity
        df['subjectivity'] = df['text'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)

        
        # Calculate the average number of words
        average_words = df['number_of_words'].mean()

        # Calculate the percentage of sentences that have WH words
        average_wh_presence = (df['are_wh_words_present'].sum() / len(df)) * 100

        # Calculate the average polarity
        average_polarity = df['polarity'].mean()

        # Calculate the average subjectivity
        average_subjectivity = df['subjectivity'].mean()

        # Display the calculated averages
        print("Average Number of Words:", average_words)
        print("Average Percentage of Sentences with WH Words:", average_wh_presence)
        print("Average Polarity:", average_polarity)
        print("Average Subjectivity:", average_subjectivity)

        # Create a DataFrame to store the results
        results_df = pd.DataFrame({
            'Metric': ['Average Number of Words', 'Average Percentage of Sentences with WH Words', 'Average Polarity', 'Average Subjectivity'],
            'Value': [average_words, average_wh_presence, average_polarity, average_subjectivity]
        })
        st.subheader("Sentiment Analysis Dataframe")
        st.table(results_df)
        
        # emo_in_txt = text
        # Define cache for the analyzers
        # from setup import analyzer, emotion_analyzer, hate_speech_analyzer
        # @st.cache(allow_output_mutation=True)
        # def create_analyzers():
        #     return analyzer, emotion_analyzer, hate_speech_analyzer
        
        # analyzers = create_analyzers()
        # sentiment1 = analyzers[0].predict(text)
        # emotion1 = analyzers[1].predict(text)
        # hate_speech1 = analyzers[2].predict(text)
        # analyzer = create_analyzer(task="sentiment", lang="en")
        # sentiment1 = analyzer.predict(text)
        st.subheader("Sentiment Analysis")
        st.write(sentiment1)
        print(sentiment1)
        sentiment_output = sentiment1.output
        probas_sentiment = sentiment1.probas
        NEU = probas_sentiment.get("NEU")
        POS = probas_sentiment.get("POS")
        NEG = probas_sentiment.get("NEG")
        

        # Create labels and values for the pie chart
        labels = ['NEU', 'POS', 'NEG']
        values = [NEU, POS, NEG]
        colors = ['blue', 'green', 'red']
        
        with _lock:
            # Create a figure with the figure number 7
            plt.figure(figsize=(6, 6))
            
            # Create a pie chart with custom colors
            wedges, _ = plt.pie(values, colors=colors, startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
            
            # Create a legend with labels and values
            legend_labels = [f"{label}: {value:.1%}" for label, value in zip(labels, values)]
            plt.legend(wedges, legend_labels, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.show()
            st.pyplot(plt, use_container_width=True)

        st.write("Sentiment Output:", sentiment_output)
        st.write("Probas Sentiment:")
        st.write("NEU:", NEU)
        st.write("POS:", POS)
        st.write("NEG:", NEG)
        
        # emotion_analyzer = create_analyzer(task="emotion", lang="en")
        # emotion1 = emotion_analyzer.predict(text)
        st.subheader("Emotion Analysis")
        st.write(emotion1)
        print(emotion1)
        emotion_output = emotion1.output
        probas_emotion = emotion1.probas
        others = probas_emotion.get("others")
        joy = probas_emotion.get("joy")
        disgust = probas_emotion.get("disgust")
        fear = probas_emotion.get("fear")
        sadness = probas_emotion.get("sadness")
        surprise = probas_emotion.get("surprise")
        anger = probas_emotion.get("anger")
        

        # Create a dictionary for the emotion probabilities
        emotions101 = {
            "Others": others,
            "Joy": joy,
            "Disgust": disgust,
            "Fear": fear,
            "Sadness": sadness,
            "Surprise": surprise,
            "Anger": anger
        }
        # Extract emotion labels and probabilities
        emotions56 = emotions101.keys()
        probabilities56 = emotions101.values()
        
        # Create a bar plot
        with _lock:
            plt.figure(figsize=(8, 4))
            plt.bar(emotions56, probabilities56, color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
            plt.xlabel("Emotion")
            plt.ylabel("Probability")
            plt.title("Emotion Probabilities")
            plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
            plt.show()
            st.pyplot(plt, use_container_width=True)

        st.write("Emotion Output:", emotion_output)
        st.write("Probas Emotion: ⤵️")
        st.write("Others:", others)
        st.write("Joy:", joy)
        st.write("Disgust:", disgust)
        st.write("Fear:", fear)
        st.write("Sadness:", sadness)
        st.write("Surprise:", surprise)
        st.write("Anger:", anger)
        # Show the plot
        
        # st.bar_chart(emotions101)
        # with _lock:
        #     plt.figure(8)
        #     plt.barh(list(emotions101.keys()), list(emotions101.values()))
        #     plt.xlabel('Probability')
        #     plt.title('Emotion Analysis')
        #     st.pyplot(plt.figure(8), use_container_width=True)
        
        # hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en")

        # hate_speech1 =  hate_speech_analyzer.predict(text)
        st.subheader("Hate Speech Analysis")
        st.write(hate_speech1)
        print(hate_speech1)
        hate_speech1_output = hate_speech1.output
        probas_hate_speech1 = hate_speech1.probas
        # Extract the values
        hateful = probas_hate_speech1.get("hateful")
        targeted = probas_hate_speech1.get("targeted")
        aggressive = probas_hate_speech1.get("aggressive")
        
     
        
        # Create a dictionary for the hate speech probabilities
        hate_speech = {
            "Hateful": hateful,
            "Targeted": targeted,
            "Aggressive": aggressive
        }
        
        # Extract hate speech labels and probabilities
        labels = hate_speech.keys()
        probs = hate_speech.values()
        
        # Create a bar plot
        with _lock:
            plt.figure(figsize=(8, 4))
            plt.bar(labels, probs, color=['red', 'green', 'blue'])
            plt.xlabel("Category")
            plt.ylabel("Probability")
            plt.title("Hate Speech Probabilities")
            plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
            # Show the plot
            plt.show()
            st.pyplot(plt, use_container_width=True)
            st.write("Hate Speech Output:", hate_speech1_output)

        st.write("Probas:")
        st.write("Hateful:", hateful)
        st.write("Targeted:", targeted)
        st.write("Aggressive:", aggressive) 
        # Now, text contains all the video transcripts as a single string

        for idx, transcript in enumerate(user_transcripts, start=1):
            st.subheader(f"Processed Transcript {idx}:")
            st.write("Transcript Text:")
            st.write(transcript)  # Display the transcript

            # Generate and display the summary
            summary = summarize_text(transcript)
            st.subheader(f"Summary for Transcript {idx}:")
            st.write(summary)  # Display the summary

            # Extract keywords from each transcript
            keywords = extract_keywords(transcript, top_n=10)
            st.subheader(f"Keywords for Transcript {idx}:")
            st.write(keywords)

        
        st.write("Note: This app uses the YouTube Transcript API to retrieve captions.")
    with col22:
        
        # import nltk
        # from nltk.corpus import stopwords
        # from nltk.tokenize import word_tokenize, sent_tokenize
        # nltk.download('stopwords')
        # nltk.download('punkt')
        # stopWords = set(stopwords.words("english"))
        # Function to summarize a given text
        # def summarize_text(text):
        #     # Tokenizing the text
        #     words = word_tokenize(text)
        
        #     # Creating a frequency table to keep the score of each word
        #     freqTable = dict()
        #     for word in words:
        #         word = word.lower()
        #         if word in stopWords:
        #             continue
        #         if word in freqTable:
        #             freqTable[word] += 1
        #         else:
        #             freqTable[word] = 1
        
        #     # Creating a dictionary to keep the score of each sentence
        #     sentences = sent_tokenize(text)
        #     sentenceValue = dict()
        
        #     for sentence in sentences:
        #         for word, freq in freqTable.items():
        #             if word in sentence.lower():
        #                 if sentence in sentenceValue:
        #                     sentenceValue[sentence] += freq
        #                 else:
        #                     sentenceValue[sentence] = freq
        
        #     sumValues = 0
        #     for sentence in sentenceValue:
        #         sumValues += sentenceValue[sentence]
        
        #     # Average value of a sentence from the original text
        #     average = int(sumValues / len(sentenceValue))
        
        #     # Storing sentences into our summary.
        #     summary = ''
        #     for sentence in sentences:
        #         if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
        #             summary += " " + sentence
        
        #     return summary

        # Function to extract common keywords from a given text
        # def extract_keywords(text, top_n=10):
        #     # Preprocess and remove stopwords
        #     filtered_words = []
        #     words = nltk.word_tokenize(text)
        #     for word in words:
        #         word = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', '', word)
        #         if word.lower() not in stopWords and len(word) > 1:
        #             filtered_words.append(word.lower())
        
        #     # Extract keywords (non-stopwords as keywords)
        #     keywords = [word for word in filtered_words]
        
        #     # Count word frequencies
        #     word_count = Counter(keywords)
        
        #     # Extract the top N keywords based on their frequency
        #     top_keywords = [keyword for keyword, _ in word_count.most_common(top_n)]
        
        #     return top_keywords
        
        user_video_urls1 = [url1 for url1 in user_urls1 if url1.strip()]
        user_transcripts1 = []

        for url1 in user_video_urls1:
            transcript1 = get_youtube_transcript(url1)

            if transcript1:
                user_transcripts1.append(transcript1)
        # Convert the list of transcripts into a single string
        text1 = "\n".join(user_transcripts1)
        # from setup import analyzer, emotion_analyzer, hate_speech_analyzer
        # from transformers import AddedToken
        
        # # Define a custom hash function for tokenizers.AddedToken
        # def my_hash_func(token):
        #     try:
        #         return hash((token.ids, token.type_id))
        #     except AttributeError:
        #         # Handle cases where the token object is not as expected
        #         return hash(str(token))
        
        # @st.cache(allow_output_mutation=True, hash_funcs={AddedToken: my_hash_func})
        # def create_analyzers():
        #     return analyzer, emotion_analyzer, hate_speech_analyzer
        
        # analyzers = create_analyzers()
        
        # sentiment11 = analyzers[0].predict(text1)
        # emotion11 = analyzers[1].predict(text1)
        # hate_speech11 = analyzers[2].predict(text1)
        # from setup import analyzer, emotion_analyzer, hate_speech_analyzer
        # from transformers import AddedToken
        
        # # Define a custom hash function for tokenizers.AddedToken
        # def my_hash_func(token):
        #     try:
        #         return hash((token.ids, token.type_id))
        #     except AttributeError:
        #         # Handle cases where the token object is not as expected
        #         return hash(str(token))
        
        # @st.cache_data(hash_funcs={AddedToken: my_hash_func})
        # def create_analyzers():
        #     return analyzer, emotion_analyzer, hate_speech_analyzer
        
        # analyzers = create_analyzers()
        sentiment11 = analyzers[0].predict(text1)
        emotion11 = analyzers[1].predict(text1)
        hate_speech11 = analyzers[2].predict(text1)

        TOInews1 = re.sub("[^A-Za-z" "]+", " ", text1).lower()
                    
        TOInews_tokens1 = TOInews1.split(" ")

        with open("en-stop-words.txt", "r") as sw:
            stop_words1 = sw.read()
            
        stop_words1 = stop_words1.split("\n")

        tokens1 = [w1 for w1 in TOInews_tokens1 if not w1 in stop_words1]

        tokens_frequencies1 = Counter(tokens1)

        # tokens_frequencies = tokens_frequencies.loc[tokens_frequencies.english_text != "", :]
        # tokens_frequencies = tokens_frequencies.iloc[1:]

        # Sorting
        tokens_frequencies1 = sorted(tokens_frequencies1.items(), key = lambda x1: x1[1])

        # Storing frequencies and items in separate variables 
        frequencies1 = list(reversed([i1[1] for i1 in tokens_frequencies1]))
        words1 = list(reversed([i1[0] for i1 in tokens_frequencies1]))

        # Barplot of top 10 
        # import matplotlib.pyplot as plt
        
        
        # Create a figure and bar chart
        with _lock:
            plt.figure(figsize=(8, 4))
            plt.bar(height=frequencies1[0:11], x=range(0, 11), color=['red', 'green', 'black', 'yellow', 'blue', 'pink', 'violet'], width=0.6)
            plt.title("Top 10 Tokens (Words)")
            plt.grid(True)
            # Customize the x-axis labels and rotation for visibility
            plt.xticks(range(0, 11), words1[0:11], rotation=45)
            plt.xlabel("Tokens")
            plt.ylabel("Count")
            
            # Display the plot in Streamlit
            st.pyplot(plt, use_container_width=True)
        ##########

        st.write("Please be patience for Amazing Results it will take few minutes")
        # Joinining all the tokens into single paragraph 
        cleanstrng1 = " ".join(words1)

        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_ip1 = WordCloud(background_color = 'White', width = 2800, height = 2400).generate(cleanstrng1)
            plt.title("Normal Word Cloud")
            plt.axis("off")
            plt.grid(False)
            plt.imshow(wordcloud_ip1)
            st.pyplot(plt, use_container_width=True)


        #########################################################################################

        # positive words

        # with open("en-positive-words.txt", "r") as pos: #IMP
        #   poswords = pos.read().split("\n")
        
        # Positive word cloud
        # Choosing the only words which are present in positive words
        # Positive word cloud
        # Choosing the only words which are present in positive words
        pos_tokens1 = " ".join ([w1 for w1 in TOInews_tokens1 if w1 in poswords])

        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_positive1 = WordCloud(background_color = 'White', width = 1800, height = 1400).generate(pos_tokens1)
            plt.title("Positive Word Cloud")
            plt.axis("off")
            plt.grid(False)
            plt.imshow(wordcloud_positive1)
            st.pyplot(plt, use_container_width=True)

        # Negative words
       
        # with open("en-negative-words.txt", "r") as neg: #IMP
        #   negwords = neg.read().split("\n")
        # Negative word cloud
        # Choosing the only words which are present in negwords
        neg_tokens1 = " ".join ([w1 for w1 in TOInews_tokens1 if w1 in negwords])
        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_negative1 = WordCloud(background_color = 'black', width = 1800, height=1400).generate(neg_tokens1)
            plt.title("Negative Word Cloud")
            plt.axis("off")
            plt.grid(False)
            plt.imshow(wordcloud_negative1)
            st.pyplot(plt, use_container_width=True)
        #########################################################################################
        
        
        # Word cloud with 2 words together being repeated

        # Extracting n-grams using TextBlob

        bigrams_list1 = list(nltk.bigrams(tokens1))
        dictionary21 = [' '.join(tup) for tup in bigrams_list1]

        # Using count vectorizer to view the frequency of bigrams
        
        vectorizer11 = CountVectorizer(ngram_range = (2, 2))
        bag_of_words11 = vectorizer11.fit_transform(dictionary21)
        v11 = vectorizer11.vocabulary_

        sum_words11 = bag_of_words11.sum(axis = 0)
        words_freq11 = [(word11, sum_words11[0, idx11]) for word11, idx11 in v11.items()]
        words_freq11 = sorted(words_freq11, key = lambda x11: x11[1], reverse = True)

        words_dict11 = dict(words_freq11[:100])
        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_21 = WordCloud(background_color = 'black', width = 1800, height = 1400)                 
            wordcloud_21.generate_from_frequencies(words_dict11)
            plt.title("Bi-Gram based on Frequency")
            plt.axis("off")
            plt.grid(False)
            plt.imshow(wordcloud_21)
            st.pyplot(plt, use_container_width=True)
        ##############################################################################################
        
        # Word cloud with 2 words together being repeated
        
        # Extracting n-grams using TextBlob

        bigrams_list22 = list(nltk.trigrams(tokens1))
        dictionary32 = [' '.join(tup) for tup in bigrams_list22]

        # Using count vectorizer to view the frequency of bigrams
        
        vectorizer12 = CountVectorizer(ngram_range = (3, 3))
        bag_of_words12 = vectorizer12.fit_transform(dictionary32)
        v22 = vectorizer12.vocabulary_

        sum_words12 = bag_of_words12.sum(axis = 0)
        words_freq12 = [(word12, sum_words12[0, idx12]) for word12, idx12 in v22.items()]
        words_freq12 = sorted(words_freq12, key = lambda x2: x2[1], reverse = True)

        words_dict12 = dict(words_freq12[:100])
        with _lock:
            plt.figure(figsize=(8, 4))
            wordcloud_32 = WordCloud(background_color = 'black', width = 1800, height = 1400)                  
            wordcloud_32.generate_from_frequencies(words_dict12)
            plt.title("Tri-Gram based on Frequency")
            plt.grid(False)
            plt.axis("off")
            plt.imshow(wordcloud_32)
            st.pyplot(plt, use_container_width=True)

        # eqn shift 1
        pattern1 = "[^A-Za-z.]+"

        # Perform english_text preprocessing without removing full stops
        sen1 = re.sub(pattern1, " ", text1).lower()

        # SENTANCE Tokenizer
        sen_t1 = sen1.split(".")


        # Create a DataFrame with the sentences as lists
        df1 = pd.DataFrame(sen_t1)

        # Display the DataFrame
        print(df1)

        df1.columns = ['english_text']
        

        # Number of words
        
        df1['number_of_words'] = df1['english_text'].apply(lambda x1: len(TextBlob(x1).words))

        # Detect presence of wh words
        wh_words1 = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])
        df1['are_wh_words_present'] = df1['english_text'].apply(lambda x1 : True if len(set(TextBlob(str(x1)).words).intersection(wh_words1)) > 0 else False)


        # Polarity
        df1['polarity'] = df1['english_text'].apply(lambda x1 : TextBlob(str(x1)).sentiment.polarity)

        # Subjectivity
        df1['subjectivity'] = df1['english_text'].apply(lambda x1 : TextBlob(str(x1)).sentiment.subjectivity)

        
        # Calculate the average number of words
        average_words1 = df1['number_of_words'].mean()

        # Calculate the percentage of sentences that have WH words
        average_wh_presence1 = (df1['are_wh_words_present'].sum() / len(df1)) * 100

        # Calculate the average polarity
        average_polarity1 = df1['polarity'].mean()

        # Calculate the average subjectivity
        average_subjectivity1 = df1['subjectivity'].mean()

        # Display the calculated averages
        print("Average Number of Words:", average_words1)
        print("Average Percentage of Sentences with WH Words:", average_wh_presence1)
        print("Average Polarity:", average_polarity1)
        print("Average Subjectivity:", average_subjectivity1)

        # Create a DataFrame to store the results
        results_df1 = pd.DataFrame({
            'Metric': ['Average Number of Words', 'Average Percentage of Sentences with WH Words', 'Average Polarity', 'Average Subjectivity'],
            'Value': [average_words1, average_wh_presence1, average_polarity1, average_subjectivity1]
        })

        # Display the results DataFrame
        print(results_df1)
        # eqn shift 1

        # results_df = pd.DataFrame(results_df)
        # Set a Seaborn color palette for styling
        
        
        # Streamlit app
        st.subheader("Sentiment Analysis Dataframe")
        
        # Display the DataFrame using Seaborn styling
        st.table(results_df1)
        # Open the file in read mode and read its content into a variable
        
        # emo_in_txt1 = text1
        
        # result = multi_emotion.predict([emo_in_txt])


        # # Extract english_text, labels, and probabilities
        # text1 = result[0]['english_text']
        # labels = result[0]['pred_label'].split(',')
        # probabilities = json.loads(result[0]['probability'])

        # # Split the english_text into words
        # words_e = text1.split()

        # # Create a list of dictionaries to store emotions for each word
        # emotions_list = []

        # for word_e, prob in zip(words_e, probabilities):
        #     emotions = {'word': word_e}
        #     emotions.update(prob)
        #     emotions_list.append(emotions)

        # print(emotions_list)


        # label_1 = [key for item in probabilities for key in item.keys()] # True

        # print(label_1) #TRUE

        # # Create a DataFrame to capture emotions for each word
        # emotions_df = pd.DataFrame(emotions_list)

        # # Print the DataFrame
        # print(emotions_df)
        # # Now you have a DataFrame with emotions for each word

        # # Assuming you have 'Happy', 'Angry', 'Surprise', 'Sad', and 'Fear' columns in emotions_df
        # # You can now perform the additional operations as requested:
        # tokens_df = pd.DataFrame(words_e, columns=['words'])

        # emp_emotions = pd.concat([tokens_df, emotions_df], axis=1)


        # emotions_to_plot = label_1
        # sum_emotions = emp_emotions[emotions_to_plot].sum()

        # # Plot the summed emotions
        # with _lock:
        #     plt.figure(7)
        #     sum_emotions.plot(kind='bar', color=['pink', 'orange', 'blue', 'yellow', 'green', 'purple', 'red', 'cyan', 'magenta', 'lime'])
        #     plt.title('Sum of Emotions for the english_text')
        #     plt.xlabel('Emotions')
        #     plt.ylabel('Sum')
        #     plt.show()
        #     st.pyplot(plt.figure(7), use_container_width=True)
        # print(emp_emotions.head(20))
        # from setup import analyzer, emotion_analyzer, hate_speech_analyzer
        # analyzer = create_analyzer(task="sentiment", lang="en")
        # sentiment11 = analyzer.predict(emo_in_txt1)
        st.subheader("Sentiment Analysis")
        st.write(sentiment11)
        sentiment_output1 = sentiment11.output
        probas_sentiment1 = sentiment11.probas
        NEU1 = probas_sentiment1.get("NEU")
        POS1 = probas_sentiment1.get("POS")
        NEG1 = probas_sentiment1.get("NEG")
        

        # Create labels and values for the pie chart
        labels1 = ['NEU', 'POS', 'NEG']
        values1 = [NEU1, POS1, NEG1]
        colors1 = ['blue', 'green', 'red']
        
        with _lock:
            # Create a figure with the figure number 7
            plt.figure(figsize=(6, 6))
            
            # Create a pie chart with custom colors
            wedges, _ = plt.pie(values1, colors=colors1, startangle=90)
            plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
            
            # Create a legend with labels and values
            legend_labels1 = [f"{label1}: {value1:.1%}" for label1, value1 in zip(labels1, values1)]
            plt.legend(wedges, legend_labels1, title="Sentiments", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.show()
            st.pyplot(plt, use_container_width=True)

        st.write("Sentiment Output:", sentiment_output1)
        st.write("Probas Sentiment:")
        st.write("NEU:", NEU1)
        st.write("POS:", POS1)
        st.write("NEG:", NEG1)
        
        # emotion_analyzer = create_analyzer(task="emotion", lang="en")
        # emotion11 = emotion_analyzer.predict(emo_in_txt1)
        st.subheader("Emotion Analysis")
        st.write(emotion11)
        emotion_output1124 = emotion11.output
        probas_emotion1124 = emotion11.probas
        others114 = probas_emotion1124.get("others")
        joy114 = probas_emotion1124.get("joy")
        disgust114 = probas_emotion1124.get("disgust")
        fear114 = probas_emotion1124.get("fear")
        sadness114 = probas_emotion1124.get("sadness")
        surprise114 = probas_emotion1124.get("surprise")
        anger114 = probas_emotion1124.get("anger")
        

        # Create a dictionary for the emotion probabilities
        emotions1011 = {
            "Others": others114,
            "Joy": joy114,
            "Disgust": disgust114,
            "Fear": fear114,
            "Sadness": sadness114,
            "Surprise": surprise114,
            "Anger": anger114
        }
        # Extract emotion labels and probabilities
        emotions111 = emotions1011.keys()
        probabilities111 = emotions1011.values()
        
        # Create a bar plot
        with _lock:
            plt.figure(figsize=(8, 4))
            plt.bar(emotions111, probabilities111, color=['blue', 'green', 'red', 'purple', 'orange', 'pink'])
            plt.xlabel("Emotion")
            plt.ylabel("Probability")
            plt.title("Emotion Probabilities")
            plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
            plt.show()
            st.pyplot(plt, use_container_width=True)

        st.write("Emotion Output:", emotion_output1124)
        st.write("Probas Emotion:")
        st.write("Others:", others114)
        st.write("Joy:", joy114)
        st.write("Disgust:", disgust114)
        st.write("Fear:", fear114)
        st.write("Sadness:", sadness114)
        st.write("Surprise:", surprise114)
        st.write("Anger:", anger114)
        # Show the plot
        
        # st.bar_chart(emotions101)
        # with _lock:
        #     plt.figure(8)
        #     plt.barh(list(emotions101.keys()), list(emotions101.values()))
        #     plt.xlabel('Probability')
        #     plt.title('Emotion Analysis')
        #     st.pyplot(plt.figure(8), use_container_width=True)
        
        # hate_speech_analyzer = create_analyzer(task="hate_speech", lang="en")

        # hate_speech11 =  hate_speech_analyzer.predict(emo_in_txt1)
        st.subheader("Hate Speech Analysis")
        st.write(hate_speech11)
        hate_speech1_output111 = hate_speech11.output
        probas_hate_speech111 = hate_speech11.probas
        # Extract the values
        hateful111 = probas_hate_speech111.get("hateful")
        targeted111 = probas_hate_speech111.get("targeted")
        aggressive111 = probas_hate_speech111.get("aggressive")
        
     
        
        # Create a dictionary for the hate speech probabilities
        hate_speech1117 = {
            "Hateful": hateful111,
            "Targeted": targeted111,
            "Aggressive": aggressive111
        }
        
        # Extract hate speech labels and probabilities
        labels1111 = hate_speech1117.keys()
        probs1111 = hate_speech1117.values()
        
        # Create a bar plot
        with _lock:
            plt.figure(figsize=(8, 4))
            plt.bar(labels1111, probs1111, color=['red', 'green', 'blue'])
            plt.xlabel("Category")
            plt.ylabel("Probability")
            plt.title("Hate Speech Probabilities")
            plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
            # Show the plot
            plt.show()
            st.pyplot(plt, use_container_width=True)
            st.write("Hate Speech Output:", hate_speech1_output111)
        
        st.write("Probas:")
        st.write("Hateful:", hateful111)
        st.write("Targeted:", targeted111)
        st.write("Aggressive:", aggressive111) 

        for idx, transcript1 in enumerate(user_transcripts1, start=1):
            st.subheader(f"Processed Transcript {idx}:")
            st.write("Transcript Text:")
            st.write(transcript1)  # Display the transcript

            # Generate and display the summary
            summary1 = summarize_text(transcript1)
            st.subheader(f"Summary for Transcript {idx}:")
            st.write(summary1)  # Display the summary

            # Extract keywords from each transcript
            keywords1 = extract_keywords(transcript1, top_n=10)
            st.subheader(f"Keywords for Transcript {idx}:")
            st.write(keywords1)

        # Extract common keywords from all user transcripts
        st.write("Note: This app uses the YouTube Transcript API to retrieve captions.")    
        st.balloons()
        # Display a horizontal bar chart for hate speech probabilities
        # st.subheader("Hate Speech Analysis")
        # st.bar_chart(hate_speech)

        ########## End ###########
        
        ##############################################################################################



        # Extracting general features from raw texts
        # Number of words
        # Detect presence of wh words
        # Polarity
        # Subjectivity
        # Language identification


        # Define the regular expression to exclude periods
        # pattern = "[^A-Za-z.]+"

        # # Perform english_text preprocessing without removing full stops
        # sen = re.sub(pattern, " ", english_text).lower()

        # # SENTANCE Tokenizer
        # sen_t = sen.split(".")


        # # Create a DataFrame with the sentences as lists
        # df = pd.DataFrame(sen_t)

        # # Display the DataFrame
        # print(df)

        # df.columns = ['english_text']
        

        # # Number of words
        # df['number_of_words'] = df['english_text'].apply(lambda x : len(TextBlob(x).words))

        # # Detect presence of wh words
        # wh_words = set(['why', 'who', 'which', 'what', 'where', 'when', 'how'])
        # df['are_wh_words_present'] = df['english_text'].apply(lambda x : True if len(set(TextBlob(str(x)).words).intersection(wh_words)) > 0 else False)


        # # Polarity
        # df['polarity'] = df['english_text'].apply(lambda x : TextBlob(str(x)).sentiment.polarity)

        # # Subjectivity
        # df['subjectivity'] = df['english_text'].apply(lambda x : TextBlob(str(x)).sentiment.subjectivity)

        
        # # Calculate the average number of words
        # average_words = df['number_of_words'].mean()

        # # Calculate the percentage of sentences that have WH words
        # average_wh_presence = (df['are_wh_words_present'].sum() / len(df)) * 100

        # # Calculate the average polarity
        # average_polarity = df['polarity'].mean()

        # # Calculate the average subjectivity
        # average_subjectivity = df['subjectivity'].mean()

        # # Display the calculated averages
        # print("Average Number of Words:", average_words)
        # print("Average Percentage of Sentences with WH Words:", average_wh_presence)
        # print("Average Polarity:", average_polarity)
        # print("Average Subjectivity:", average_subjectivity)

        # # Create a DataFrame to store the results
        # results_df = pd.DataFrame({
        #     'Metric': ['Average Number of Words', 'Average Percentage of Sentences with WH Words', 'Average Polarity', 'Average Subjectivity'],
        #     'Value': [average_words, average_wh_presence, average_polarity, average_subjectivity]
        # })

        # # Display the results DataFrame
        # print(results_df)
        
        # # Display the plots
        # st.pyplot(plt.figure(1), use_container_width=True)  # Figure 1
        # st.pyplot(plt.figure(2), use_container_width=True)  # Figure 2
        # st.pyplot(plt.figure(3), use_container_width=True)  # Figure 3
        # st.pyplot(plt.figure(4), use_container_width=True)  # Figure 4
        # st.pyplot(plt.figure(5), use_container_width=True)  # Figure 5
        # st.pyplot(plt.figure(6), use_container_width=True)  # Figure 6
        # st.pyplot(plt.figure(7), use_container_width=True)  # Figure 7
        
        
        # for idx, news_item in enumerate(user_news_data, start=1):
        #     st.subheader(f"News Article {idx}: {news_item['Title']}")
        #     st.write("Article Text:")
        #     st.write(news_item['Article Text'])
        #     st.write("Article Summary:")
        #     st.write(news_item['Article Summary'])
        #     st.write("Article Keywords:")
        #     st.write(', '.join(news_item['Article Keywords']))
        #     st.markdown("---")    
        # results_df = pd.DataFrame(results_df)
        # # Set a Seaborn color palette for styling
        
        
        # # Streamlit app
        # st.subheader("Sentiment Analysis Dataframe")
        
        # # Display the DataFrame using Seaborn styling
        # st.table(results_df)

        st.write("Note: Each user's data is stored in-memory and not shared with others.")

