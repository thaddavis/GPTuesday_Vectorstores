Focus

In this video we’ll do a quick dive on (cut) “Vectorstores” aka “Vector Databases” aka “Vector Embeddings” 
If you’ve ever heard the term “embeddings”, or “word embeddings”, or wondered what (cut) that section on OpenAI’s site labelled “Embeddings” is all about, then this video is for you…  If you’ve never heard of “Vectorstores” or “embeddings” than that’s ok because you’ve likely interacted with them unknowingly…  Some common consumer-facing applications powered by Vectorstores would be… 
(Cut)
 1) Anomaly detection related to your bank account 2) Recommendations for content on streaming platforms 3) Automated content moderation on social media  Prereqs…

The only prerequisites you need to know in order to follow along with this video are the concepts of…
 
 1) a vector  2) a dot product

(cut) A vector is a multidimensional value. This is in contrast to a 1 dimensional value (which is sometimes referred to as a scalar)  (look )For example, if you were to ask me (cut) “What’s my age again?” *(look) I might say “33” and that would be an acceptable 1-dimensional response. But now let’s say you asked me (cut) “How would you throw this basketball so that it goes into the hoop?”, (look) I then might say that I would launch it at this angle + at this speed + in this direction…  (cut) You see how there are multiple dimensions involved with the value that answers this second question? (Look) Multi-dimensional values are what vectors represent…   Many things can be represented with vectors, for example, let’s represent a person’s personality with a 4-dimensional vector…  (cut) We’ll take the 1st number to be how confident they are, the 2nd how social they are, the 3rd how analytical they are, and the 4th how trusting they are.  (look) Hopefully that gives a good understanding of what a vector is…

(cut) A dot product is a math formula just like “mx + b = y” or “a2 + b2 = c2” except that it takes 2 vectors as an input and spits out a value telling how similar or different the 2 vectors are…  The more positive (+) the output of the dot product, the more “similar” the 2 vectors are considered to be… The more negative (-) the output of the dot product, the more “opposite” they are… - if the dot product outputs a value of 0 then that means the vectors are “unrelated” ie: “orthogonal” ie: they are neither similar nor different
 (look) Ok!  The rest of this video will comprise of 3 parts  (cut) PART 1 is a rip off of Dave Shapiro’s video entitled “What the heck are embeddings?” and will lay the ground for the subsequent 2 parts 
PART 2 is a demo of how to use a “Vector Database” aka a “Vectorstore” (shoutout to Chris Johnston for the tips and tricks I’ll be showing here)  PART 3 is a very quick rundown of two more advanced concepts related to Vector Databases (shoutout to Ryan Lambert for prompting me to cover these)

—————————————— DEMO 1 ——————————————— (FULL SCREEN w/ sound effect)

(look) For this demo we’ll be using Python + an “embedding” model from OpenAI…  While most people have heard of ChatGPT and GPT4 there are other models OpenAI offers as well…  (cut) For example, “whisper” is an OpenAI model trained to receive audio and spit out transcriptions…  DALL-E,  is a model that receives text and spits out images that represent the text..  the “embedding” model we’ll be using is called `text-embedding-ada-002`  it’s a model that receives text and spits out a VECTOR with 1536 dimensions…  (look) I looked into what the dimensions of these output vectors mean but didn’t find anything  If anyone knows, please let me know : )   (cut) I’m guessing the 1st component means how masculine/feminine the original text is, the 2nd how scientific/unscientific the text is, the 3rd how Western/Eastern the text is I don’t know…  (look) Hopefully that gives a gist of what’s happening…  (cut) So, if we send the word “tiger” to `text-embedding-ada-002` this is the vector we get back…  Or let’s send the word “cat”…

here is the vector we get back for the word “cat”…  Now let’s send the word “fish” 
 (look) Here is where the dot product comes into play…

(Cut) Let’s produce the dot product of the vectors we received for “tiger” and “cat” and compare it against the dot product of the vectors we got for “tiger” and “fish”…  You can see that the value for “tiger”/“cat” is higher than the value of “tiger”/“fish” and we can interpret that to mean that “tiger”/“cat” are more similar conceptually than “tiger”/“fish” is  Or let’s compare the vectors for “Monday”, “Tuesday”, “Wednesday”  The dot product for “Monday”/“Tuesday” has higher dot value then the dot product for “Monday/'Wednesday”  (look) Now you see why the word “embeddings” is used to refer to these vectors. It’s because the models that generate these vectors embed notions of meaning into them…

(cut) I couldn’t come up with a way to visualize 1536 but here is a visualization of a collection of 3-dimensional embedding vectors…  https://projector.tensorflow.org/  Each vector is represented as a point and the points that are closer together other represent similar concepts  For example if we look over here we see…  Or over here…  END DEMO 1  —————————————— DEMO 2 ———————————————  For this demo we’ll take a closer look at the “Vector Database” side of things  Just like SQL databases store data into uniformly labelled tables… and no-SQL databases store collections of objects or dictionaries… Vector databases store vectors…

At the time of recording in 2023, there are many vector databases on the market for example Pinecone, Chroma, or Milvus  and all of them are valid, but for this demo we’ll use Chroma simply because it’s FREE…  Running ChromaDB is a 3-step process and only requires Docker to be installed on your host machine… 
The steps for installing ChromaDB are…
 1) Clone the .git repository 2) Build the image 3) Run a container based on the image

If everything works you should see a ChromaDB server running in your console…  (look) Two common uses for “Vector Databases” or “Vectorstores” would be “similarity search” and “anomaly detection”

From this list of common consumer-facing applications of vectorstores, content moderation on social media would be an example of similarity search…  With “similarity search”, you are comparing vectors that represent user data against vectors that represent concepts defined as unacceptable by a given platform and where you find the similarity to cross a certain threshold, you will mark the relevant user data for hiding or deletion.  Unusual activity associated with your bank account on the other hand is an example of anomaly detection and involves generating vectors for all the actions you perform against your bank account and raising a flag when a particular vector crosses a certain threshold of dissimilarity.  The general technique for how to do “similarity search” or “anomaly detection” yourself is outlined here… 
1 - Break up your data into chunks (for example if your data is text common choices would be by paragraph, sentence, or word) *you’ll need to tune the size of each chunk to your use-case
2 - Generate an embedding vector for each chunk in your data set using an embedding model like `text-embedding-ada-002`
3 - Convert your query to a vector using the same embedding model used in step 2 and
 if you’re performing a similarity search, return the chunks associated with the vectors that score the “highest” in similarity to the query chunk or
if you’re performing anomaly detection, you compare your query against the vectors for each chunk and raise a flag if the query vector scores below a certain threshold 
—

Let’s take a look at this list of skills that COMMAND offers…  If we search for “Mobile Development”, let’s see what gets returned… 
We can see we get matches even though the text “Mobile Development” is nowhere to be found in this document  Let’s now search for “AWS” and we can see that we get matches of various AWS services even though they’re not exactly matching the string we’re looking for…  If you look closely, you see the top results score lower and this is inconsistent with what we were seeing before with the dot product…  This is because the dot product is ONE of several formulas for measuring the similarity of 2 vectors…

ChromaDB use a distance formula instead of the Dot product which is why the higher results are showing with a lower number and the exact matches are showing as 0. A lower score means closer in distance ie higher in similarity and a score of zero means you’re query is identical to the matched chunk 
Here’s a quick overview of the code
 1 - breaking things up either by spaces or commas on line 56 2 - generating vectors for each chunk on lines 85-88 3 - searching for the top 10 results that match our query on lines 96-99
 https://towardsdatascience.com/similarity-search-knn-inverted-file-index-7cab80cc0e79

—————————————— DEMO 3 (not really a demo) ——————————————— 
Let’s talk about “indexing”  Indexing involves organizing data into a format that speeds up future access or analysis  Indexing becomes necessary when searching over larger datasets regardless of whether your data in stored in an SQL, no-SQL, or vector database…  (cut) Here’s a visualization of a collection of vectors numbering in the tens of thousands and you can imagine in the case of performing similarity search against this collection, you would be waiting an increasing amount of time to search for your match as the amount of vector grows into the hundreds of thousands or millions…  The workaround for this is to group your vectors into clusters called “Voronoi Regions” and then calculate the vector in each region that is closest to all the others ie: you calculate the regions centroid

https://towardsdatascience.com/similarity-search-knn-inverted-file-index-7cab80cc0e79  Then when you perform you search against the cluster you will scan the vector space by comparing against only the centroids of each region and  not the individual vectors within each region  Once you find the centroid closest to your query you then compare against the other vectors in that region to find your closest match…  This technique is called “Inverted file index”,  While Inverted file index may not give you the best match, in practice it achieves a good tradeoff between accuracy and speed in cases where speed is more important.  Another approach for speeding up search against a large amount of vectors is to approximate the vector space by reducing or compressing the amount of dimensions of the vectors. This technique is called “Product Quantization”.

https://www.youtube.com/watch?v=iGO12Z5Lw8s  (look) If my suspicions are correct, Vector Search + an indexing technique like “Inverted file index” or “Product Quantization” is what’s behind the recommendation algorithm used in production at Spotify…

Please correct me if I am wrong… 
