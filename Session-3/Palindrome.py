#If string is a palindrome, then string should be the same if read from front or in reverse

#Function which return reverse of a string 
def Reverse(s):
    #Syntax-> [Start:End:Step]
        #'-1' means in reverse order
    return s[::-1]


def isPalindrome(s): 
	# Calling reverse function 
	rev_text = Reverse(s) 

	# Checking if both string are equal or not 
	if (s == rev_text): 
		return True
	return False


 
text = input("Enter the string to check whether it is a palindrome or not : ")
# Converting string uniformly into upper case to avoid ambiguity
text = text.upper()
# Calling Palindrom function defined
ans = isPalindrome(text) 

# 'ans' holds boolean
if ans: 
	print("\nYes, {} is a palindrome" .format(text)) 
else: 
	print("\nNo, {} is not a palindrome" .format(text))
