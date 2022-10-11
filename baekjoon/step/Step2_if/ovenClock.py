#2525 오븐 시계

#입력 - 첫째 줄 : 현재 시간 (시A 분B), 둘째 줄: 요리 필요 시간
#출력 - 결과 시간(시 분)
A, B = map(int,input().split())
C = input() #0< C <1000

C=int(C)

#C를 시 분 형식으로 변형 후 연산, 만약 분이 60이상이면 시+1이후 시가 24이상이면 -24
C_Hour = int(C/60)
C_minute = C%60

A = A+C_Hour

B = B + C_minute
if(B>=60):
    B=B-60
    A=A+1

if(A>=24):
    A=A-24

print(A,B)