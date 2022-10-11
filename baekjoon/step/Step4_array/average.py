#1546 평균

#점수/최댓값*100

#입력 - 첫줄 : 과목 개수 N,둘째 줄 : 점수 N개
#출력 - 평균

score_list = []

import sys
N = sys.stdin.readline().strip()
score_list.extend(map(int, input().split())) #list추가 extend

max_score = max(score_list)
result = 0

for i in score_list:
    result += (i/max_score)*100

print(result/int(N))