"""
site_main.js からチャット・着席関連のコードを抽出する。
"""
import re

data = open(
    r'c:\Users\FMV\Desktop\application\programming\project\PuyotanAI\site_main.js',
    encoding='utf-8'
).read()

# "sendChat" の前後 300 文字を取得
print("=== sendChat 呼び出し箇所 ===")
for m in re.finditer(r'.{0,200}sendChat.{0,200}', data):
    print(m.group())
    print("---")

# joinGame の前後を取得
print("\n=== joinGame 呼び出し箇所 ===")
for m in re.finditer(r'.{0,200}joinGame.{0,200}', data):
    print(m.group())
    print("---")

# chats コレクション書き込み箇所
print("\n=== chats コレクション書き込み ===")
for m in re.finditer(r'.{0,150}chats.{0,150}', data):
    print(m.group())
    print("---")

# e.text の近傍
print("\n=== e.text / chat rendering ===")
for m in re.finditer(r'.{0,100}e\.text.{0,100}', data):
    print(m.group())
    print("---")
