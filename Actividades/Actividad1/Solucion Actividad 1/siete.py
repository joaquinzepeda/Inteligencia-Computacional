def siete():
    i=1
    while i<=100:
        if i%7==0 or any(chr=='7' for chr in str(i)):
            print("clap ", end="")
        else:
            print(str(i)+" ", end="")
        i+=1
if __name__=="__main__":
    siete()
