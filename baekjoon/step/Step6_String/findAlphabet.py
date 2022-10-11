#10809 알파벳 찾기

#

#입력 - 소문자 영단어 S
#출력 -알파벳이 단어에 포함되어 있다면 처음 등장하는 위치, 포함되지 않으면 -1


alphabet = "abcdefghijklmnopqrstuvwxyz"
word = input()

word_list = []

for i in alphabet:
    #print(word.find(i))
    word_list.append(word.find(i))

print(word_list)