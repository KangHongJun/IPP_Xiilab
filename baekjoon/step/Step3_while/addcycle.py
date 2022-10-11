#1110 아래의 패턴으로 같은 수 찾을때 까지
# 2+6 = 8 -> 68/ 6+8 = 14 -> 84
# 8+4 = 12 ->42/ 4+2 = 6 -> 26
import sys

N = sys.stdin.readline()
N=int(N)
result = 0
num=0

if N < 10:
    N = N*10 + N

X = str(N)

while True:
    add = 0
    if N==0:
        print(1)
        break
    else:
        for i in range(len(X)):
            add += int(X[i])
        result = X[1]+str(add%10)
        X=str(result)
        num +=1
        if str(N) == str(X):
            print(num)
            break










