#N2884 - 알람 시계

#원래 설정된 알람보다 45분 일찍,  H M입력
H, M = input().split()

H=int(H)
M=int(M)

if(M<45):
    if(H==0):
        H=24
    H=H-1
    M=M+15
    print(H,M)
else:
    M=M-45
    print(H,M)