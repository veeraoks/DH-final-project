# dh-final-project

Star Wars sentiment analysis

This project examines the original Star Wars trilogy movie scripts, i.e. Star Wars, The Empire Strikes Back and Return of the Jedi. More specifically the project focuses on the sentiment of the dialogue of the most talkative characters appearing in the trilogy. The sentiment analysis was produced with SentiWordNet. Additionally, RAW Graphs was used for visualization purposes. This project aims to answer the following humanities questions:

1.	What is the overall speech sentiment of the most talkative characters in the original Star Wars trilogy? Do they express more positive or negative sentiment? Or do they remain mainly neutral?
2.	Does the sentiment of the speech of the characters change within the trilogy?

Data

As mentioned, the data used in this project is the dialogue from the original Star Wars trilogy. The data was gathered from Kaggle.com and was in tabular form where the dialogue is assigned to the character delivering the line. The data was not annotated in any way. 

Processing

The project was produced in Jupyter Notebooks and processed using Python. After loading the data into Jupyter as a pandas dataframe (a separate dataframe for each movie), the most talkative characters were determined and the top three were chosen for a more detailed analysis. The most talkative characters were determined according to the number of lines they have in the trilogy. For a more detailed analysis, one could have considered the length of their lines as well but for the exploratory nature of this project, I chose the more straightforward option. Here a judgement call was made and the top three characters were determined to be Luke, Han and C-3PO as Ben dies in Episode IV and Leia comes along so late that she is not in the top three in the first movie and as C-3PO is still in the top four in every movie.

Then, new dataframes containing the dialogue of only the top characters were created. The dialogue was roughly cleaned from excessive punctuation and stopwords as stopwords they do not carry any weight when calculating polarization with a sentiment analyzer. The stopwords were manually checked too in order to see whether they contained any words that might in this context carry any sentiment weight. The next step was to tokenize and tag the data with NLTK tagger. As the tags produced by the NLTK tagger differ from those in SentiWordNet (i.e. the tool used for the sentiment analysis), the tags needed to be transformed to match the tags in SentiWordNet. 

Next, the first synonym senses were retrieved from SentiWordNet for each word in the data. The sentiment scores were calculated according to these retrieved senses. The final sentiment for each word was determined to be either positive, negative or neutral. The overall sentiment of each character was determined based on the entire trilogy as the sentiments did not change within the trilogy. The percentages of positive, negative and neutral sentiments from the entire dialogue for each character were calculated rather than focusing on the actual sentiment scores as the movies and amount of dialogue differ among each movie and character. The final step was to create yet another dataframe to be exported and used for visualization. 

Analysis and Biases

After processing the data and examining the results, it is noticeable that SentiWordNet is not the best possible tool for sentiment analysis for speech. Mainly because the analysis was done for each word separately and thus context got ignored. This then leads for instance to the lost of sarcasm and irony that are extremely prevailing in speech. For this reason, the results mainly show a neutral tone with each character and a little difference among characters or within the trilogy can be detected. However, C-3PO does seem to express a slightly more negative tone then the two other and this could reflect the more expressive and scared nature of the character, but this conclusion would require further study with a more accurate analyzer. Furthermore, the accuracy of the tool used could have been tested. But as mentioned above, the purpose of this project was purely exploratory of the data and my skills with managing data computationally too.

Acknowledgments

Xavier. (2018). Star Wars Movie Scripts, Version 2. Retrieved November 2019 from https://www.kaggle.com/xvivancos/star-wars-movie-scripts/version/2.
