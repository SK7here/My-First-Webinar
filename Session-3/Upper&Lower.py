# Count number of upper and lower case letters

def upperlower_conventional(string): 
  
    upper = 0
    lower = 0
    others = 0
  
    for i in range(len(string)): 
          
        # For lower letters
        # ord -> Ordinal
        if (ord(string[i]) >= 97 and
            ord(string[i]) <= 122): 
            lower += 1
  
        # For upper letters 
        elif (ord(string[i]) >= 65 and
              ord(string[i]) <= 90): 
            upper += 1

        else :
            others += 1
            
  
    print('Lower case characters = %s' %lower, 
          'Upper case characters = %s' %upper,
          'Other Characters = %s' %others)



string = 'GeeksforGeeks is a portal for Geeks'
upperlower_conventional(string)

string = string.upper()
upperlower_conventional(string)

string = string.lower()
upperlower_conventional(string)




def upperlower_short(s):
    upper = 0
    lower = 0
    others = 0
    
    for i in string: 
          
        # For lower letters 
        if (i.islower()): 
            lower += 1
  
        # For upper letters 
        elif (i.isupper()): 
            upper += 1

        else :
            others += 1
  
    print('Lower case characters = {}' .format(lower), 
          'Upper case characters = {}' .format(upper),
          'Other Characters = {}' .format(others))  


 
string = 'GeeksforGeeks is a portal for Geeks'
upperlower_short(string)

string = string.upper()
upperlower_short(string)

string = string.lower()
upperlower_short(string)
