#10818 최대 최소
import sys

def get_min(N,list):
        min = list[0]
        for i in range(int(N)):
                if int(min) > int(list[i]):
                        min = list[i]
        return min

def get_max(N,list):
        max = list[0]
        for i in range(int(N)):
                if int(max) < int(list[i]):
                        max = list[i]
        return max

N = sys.stdin.readline()
A = sys.stdin.readline().split() #list(map(int,input().split()))

min = get_min(N,A) #min(A)
max = get_max(N,A) #max(B)

print(min +" "+ max)








