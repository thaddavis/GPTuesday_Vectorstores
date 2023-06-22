import openai
import json
import numpy as np

def similarity(v1, v2):
    return np.dot(v1, v2)

def main():
    print('--- Hello! ---')

    ## Example 1
    # get the vector that represents the input in the text-embedding-ada-002 model by OpenAI
    response1 = openai.Embedding.create(
        input="uzbekistan",
        model="text-embedding-ada-002"
    )
    
    print('vector embeddings returned:', len(response1['data']))
    print(len(response1['data']))
    vector_1 = response1['data'][0]['embedding']
    print('dimensions of vector', len(vector_1))

    with open('text_example_1.json', 'w') as f:
        json.dump(response1['data'], f)

    # Example 2
    # get the vector that represents the input in the text-embedding-ada-002 model by OpenAI
    response2 = openai.Embedding.create(
        input="tajikistan",
        model="text-embedding-ada-002"
    )
    
    print('vector embeddings returned:', len(response2['data']))
    print(len(response2['data']))
    vector_2 = response2['data'][0]['embedding']
    print('dimensions of vector', len(vector_2))

    with open('text_example_2.json', 'w') as f:
        json.dump(response2['data'], f)

    ## Example 3
    # get the vector that represents the input in the text-embedding-ada-002 model by OpenAI
    response3 = openai.Embedding.create(
        input="asdfasdf",
        model="text-embedding-ada-002"
    )
    
    print('vector embeddings returned:', len(response3['data']))
    print(len(response3['data']))
    vector_3 = response3['data'][0]['embedding']
    print('dimensions of vector', len(vector_3))

    with open('text_example_3.json', 'w') as f:
        json.dump(response3['data'], f)


    ## get dot product of example 1 and 2

    print(similarity(vector_1, vector_2)) # GPTuesday & GPTWednesday should score highest
    print(similarity(vector_1, vector_3)) # GPTuesday & GPTFriday should score less
    # print(similarity(vector_2, vector_3)) # GPTWednesday & GPTFriday should score somewhere in the middle

    ## get dot product of example 1 and 3