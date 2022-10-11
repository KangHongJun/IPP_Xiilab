#N14681 - 사분면 판별
x= input()
y= input()

x=int(x)
y=int(y)

if(x>0 and y >0):
    print("1")
elif(x<0 and y >0):
    print("2")
elif (x < 0 and y < 0):
    print("Step2_3")
elif(x > 0 and y < 0):
    print("4")

