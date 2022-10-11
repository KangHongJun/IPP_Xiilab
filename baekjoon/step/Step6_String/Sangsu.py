#2908 상수

#734, 893이라하면 437, 398이라고 해석하여 둘 중 큰 수를 437이라고 한다.
#input - A, B(같지 않은 3자리수) output 위의 상황에서 큰 수

N = input().split(" ")

A=N[0]
B=N[1]

reverse_A=""
reverse_B=""

for i in range(0,len(A)):
    reverse_A = A[i]+reverse_A

for i in range(0,len(B)):
    reverse_B = B[i]+reverse_B

if(int(reverse_A)>int(reverse_B)):
    print(reverse_A)
else:
    print(reverse_B)