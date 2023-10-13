# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# wow this is my first time using it 


"""
- py stands for the Python File

- To run all the codes, hit the "PLAY" button on top

- fn + 9: Run the selected code

- fn + 5: Run the entire code

- cmd + 1: # 

"""

# Create a function that computes the jaccardian distance measure between two input corpuses

def jaccardian(string1, string2):
    jac = None
    try: 
        set1 = set(string1.lower().split())
        set2 = set(string2.lower().split())
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        jac = len(intersection) / len(union)
    except: 
        print("Either one of the strings is not string type")
        pass
    return jac
    
val = jaccardian("this is a test", "so is this")
val

jaccardian("omg you stop! Bitch", "Bitch you stop!")
#jaccardian(3, "Bitch you stop!")



# for loop 
my_dictionary = dict()

my_list = [1, 5, 6, 8, 56, 3, 43]
for i in my_list:
    my_dictionary[i] = i*2
print(my_dictionary)
    
    
'''
create a function that takes in a string and outputs a dictionary where each 
unique word are the keys, and the value is the number of times the unique word
shows
'''    


input_str = 'oh my god'
dictionary_a = {}
tmp = input_str.split()
for i in set(tmp):
    dictionary_a[i] = tmp.count(i)
    
    
print(dictionary_a)

# inline loop statement 
dictionary_a = {word: input_str.split().count(word) 
                for word in set(input_str.split())}



def word_counts(string):
    dictionary_a = {}
    tokens = string.split(" ")
    for i in set(tokens): # oh, my, god, etc. 
        count = tokens.count(i)    
        dictionary_a[i]= count
    return dictionary_a

word_counts("Oh my god you are sure")

input_str = 'oh my god'
new_list = list()
for word in set(input_str.split()): 
    new_list.append(len(word))
print(new_list)


new_list = [len(i) for i in set(input_str.split())]



# while loop
cnt = 0 
while cnt < 10:
    print(cnt)
    cnt += 1
    

# Pandas API Reference 
import pandas as pd 
input_str = 'oh my god oh'
my_pd = pd.DataFrame()
tokens = input_str.split()
for word in set(tokens):
    tmp = pd.DataFrame({"word": word,
                        "frequency": tokens.count(word)}, index=[0])
    #my_pd = pd.concat(tmp)
    my_pd = pd.concat([my_pd, tmp])
print(my_pd)    
    

# Make a function 
def letsmake(string: str):
    tokens = string.split()
    df = pd.DataFrame()
    for word in set(tokens):
        new_df= pd.DataFrame({"word": word,
                              "frequency": tokens.count(word)}, index=[0])
        df= pd.concat([df, new_df])
    return df

letsmake("I am David Lee")        

# How to dump this data frame to csv 
my_pd.to_excel('Pandas.xlsx', sheet_name = "Data", index=False)    
read_data = pd.read_excel('Pandas.xlsx', sheet_name="Data")

    
    
my_pd.index
my_pd.columns
list(my_pd.columns)
my_pd["word"]
my_pd.word
    
pd_slice = my_pd[my_pd["frequency"] >= 2]
pd_slice = my_pd[(my_pd["frequency"] == 1) | (my_pd["frequency"] == 2)]








