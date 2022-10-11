#8958 OX퀴즈

#연속으로 정답이면 점수가 가산됨 ex)OOOXO = 1 2 3 0 1
#입력 - 첫 줄 : 개수, 둘째 줄 이후 : OXOX와 같은 결과
#출력 - 각 퀴즈 점수
import sys

N = input()
N=int(N)

score_list = []
for i in range(N):
    Quiz_result = []
    score = 0
    sc = 0
    Quiz_result.extend(sys.stdin.readline().strip())
    for answer in Quiz_result:
        if(answer=='O'):
            sc += 1
            score += sc
        elif(answer=='X'):
            sc = 0
    score_list.append(score)

for i in score_list:
    print(i)
