import requests
from bs4 import BeautifulSoup

# ウェブサイトのURL
url = "https://gogakunekoblog.com/english-most-common-words-1000/"

# ウェブサイトのコンテンツを取得
response = requests.get(url)
html = response.text

# BeautifulSoupを使ってHTMLをパース
soup = BeautifulSoup(html, "html.parser")

# <li>タグの中にある英単語を抜き出す
words = []
li_tags = soup.find_all("li")
for li in li_tags:
    word = li.get_text(strip=True)
    words.append(word)

# 結果を出力
print(words)
