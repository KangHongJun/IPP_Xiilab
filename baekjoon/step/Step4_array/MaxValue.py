#2562 - 9개 값 입력받아 최댓값과 최댓값의 위치 반환
import sys

A=[]
for i in range(9):
    A.append(int(sys.stdin.readline()))

Max = max(A)
print(Max)
print(A.index(Max)+1)