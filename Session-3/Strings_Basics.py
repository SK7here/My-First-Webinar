#Creating and initializing a string variable
g="Hello, Greetings to everyone gathered"
print("Sentence used for explaining basic String operations is")
print(g)


#Finding length of string
print("\n\nLength of the string is")
print(len(g))


#To crop a section of string and print
print("\n\nCropping the word 'everyone' from the string by specifying index range")
    #NOTE:Range will consider only till "ENDPOINT-1" element
print(g[20:28])
print("\n\nAccessing last character")
    #NOTE:Negative Indexing starts from end of string
print(g[-1])





#Printing the string in uppercase
print("\n\nUppercase of given string")
print(g.upper())


#Printing the string in lowercase
print("\n\nLowercase of given string")
print(g.lower())

#capitalizes first letter
print("\n\nAfter capitalizing first letter of the string")
print(g.capitalize())




#Counting the occurences
print("\n\nNumber of occurences of the letter 'e' in the string is")
print(g.count('e'))

#Returning the starting index of a word in the string
print("\n\nStarting index of word 'Hello' in the string is")
print(g.find("Hello"))




#splitting based on spaces
g=g.split(' ') 
print("\n\nSplitting string with SPACE as the delimiter")
print(g)

#Remove spaces at the end or beginning
g = " Hello "
g=g.strip()
print("\nRemoving Any white spaces in beginning or at the end")
print(g)





# Remove a character - Replace()
g = "Helololo"
print("\nRemoving first occurence of 'l' in string")
print(g.replace("l", "", 1))
print("Removing all occurences of 'l' in string")
print(g.replace("l", ""))
