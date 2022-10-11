#25304 영수증

#입력 - 첫 줄: 총 금액 X, 둘째 줄 : 물건 종류 수 N
#이후 N개의 줄 : 가격a 개수b
#출력 - 총금액과 물건들의 가격합이 같으면 Yes else No

result = 0

import sys
X = sys.stdin.readline()
X=int(X)

N = sys.stdin.readline()
N=int(N)

for i in range(N):
    a, b = sys.stdin.readline().split()
    result = result + (int(a)*int(b))

if(X==result):
    print("Yes")
else:
    print("No")
