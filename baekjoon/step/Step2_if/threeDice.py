#2480 주사위 세개


#1.같은 눈3 - 10000 + (같은 눈)x1000
#2.같은 눈2 - 1000 + (같은 눈)x100
#3. 모두 다른 눈 - (가장 큰 눈)x100

#입력 - 주사위 3개의 값
#출력 - 상급
price = 0

A,B,C = map(int,input().split())

if(A==B==C):
    price = 10000+(A*1000)
elif (A==B or A==C):
    price = 1000+(A*100)
elif (B==C):
    price = 1000+(B*100)
else:
    price = max(A,B,C)*100

print(price)