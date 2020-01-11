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

After processing the data and examining the results, it is noticeable that SentiWordNet is not the best possible tool for sentiment analysis for speech. Mainly because the analysis was done for each word separately and thus context got ignored. This then leads for instance to the lost of sarcasm and irony that are extremely prevailing in speech. For this reason, the results mainly show a neutral tone with each character and a little difference among characters or within the trilogy can be detected. However, C-3PO does seem to express a slightly more negative tone then the two other and this could reflect the more expressive and scared nature of the character, but this conclusion would require further study with a more accurate analyzer. 

Looking at the examples from the sentiments() function it can be noted that SentiWordNet is not the desirable tool for sentiment analysis for speech (or in this case fictitious speech). For instance, in the dialogue whenever the word force is mentioned, SentiWordNet assigns a neutral score to the word when in this context it should be mainly positive and in some cases negative depending on who utters the word and in what context. Furthermore, words like emperor would be considered neutral in almost any other context but in the present data, emperor should raise negative connotation and thus have a negative score if the analysis would be accurate. Additionally, verbs such as destroy, battle and destruct would be expected to have a higher negative score which is not the case. 

Three lines from Han were chose to be examined in more detail and to confirm the intuition about the non-existent impact of context. Han was chosen as he portrays a clearly a more sarcastic character than the two others. Two of the lines chosen contained a sarcastic use of the word great and the third one was a clear example of sarcasm. These were:

1.	 "Yeah, great at getting us into trouble." from Episode IV
2.	“Don't everyone thank me at once.” from Episode IV
3.	"Nice work. Great, Chewie! Great! Always thinking with your stomach." from Episode VI

In the third line, Han is not actually giving Chewie compliments rather accusing him of getting them both in trouble. The analyzer gives mainly neutral scores where more negative scores should be present. For instance, nice and work are both analyzed as neutral whereas appearing together they clearly construct a sarcastic phrase. Also, in line one, the word great should have a negative connotation as it adds to the ability to get into trouble. Thus, context does not seem to have any affect here as the words have the same score no matter where they appear in the data. 

Although these three lines were all short, they represent clear sarcasm even without any additional lexical context. However, what in my opinion is needed to detect the sarcasm straightforwardly from these quotes is at least some knowledge about the Han Solo character and the context of the movie which obviously does not transmit to the tool used. Additionally, it should be mentioned that it is impossible to account for context when the input is individual words. A more accurate analysis would then require training data as well as a better suitable tool for analyzing speech sentiment.

Furthermore, the accuracy of the tool used could have been tested. But as mentioned above, the purpose of this project was purely exploratory of the data and my skills with managing data computationally too.

Acknowledgments

Xavier. (2018). Star Wars Movie Scripts, Version 2. Retrieved November 2019 from https://www.kaggle.com/xvivancos/star-wars-movie-scripts/version/2.
