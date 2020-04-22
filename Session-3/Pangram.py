# Check if the string is pangram
# A string is said to be a pangram, if it has all the english alphabets atleast once

def ispangram(str): 
	alphabet = "abcdefghijklmnopqrstuvwxyz"
	for char in alphabet:
                # Converting string uniformly into lower case to avoid ambiguity
		if char not in str.lower(): 
			return False

	return True
	
 
string = 'The Quick Brown Fox Jumps Over The Lazy Dog'
if(ispangram(string) == True): 
	print("Yes") 
else: 
	print("No") 
