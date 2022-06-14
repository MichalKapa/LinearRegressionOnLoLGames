import numpy as np
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

###########################################################################
#
# url = "https://www.leagueofgraphs.com/summoner/champions/kassadin/eune/Micha%C5%82ek450"
#
# # dodany nagłówek, żeby strona dała sie zescrapować (urllib.error.httperror: http error 403: forbidden)
# request_site = Request(url, headers={"User-Agent": "Mozilla/5.0"})
# page = urlopen(request_site)
# html = page.read().decode('utf-8')
# soup = BeautifulSoup(html, 'html.parser')
#
# winLose = soup.find_all('div', class_='victoryDefeatText')[::2]
# kills = soup.find_all('span', class_='kills')[1::]
# deaths = soup.find_all('span', class_='deaths')[1::]
# assists = soup.find_all('span', class_='assists')[1::]
# csAndKP = soup.find_all('span', class_='requireTooltip')
#
# # usuwam zepsutą daną
# del winLose[18]
# del kills[18]
# del deaths[18]
# del assists[18]
# del csAndKP[36]
#
# cs = csAndKP[::2]
# KP = csAndKP[1::2]
#
# result = []
#
# for i in range(len(winLose)):
#     if winLose[i].text == "Victory":
#         result.append(1)
#     else:
#         result.append(0)
#
# data = []
#
# for i in range(len(winLose)):
#     data.append((result[i],
#                 int(kills[i].text),
#                 int(deaths[i].text),
#                 int(assists[i].text),
#                 int(cs[i].text.strip().replace(" CS", "")),
#                 int(KP[i].text.strip().replace("% Kills P.", ""))))
#
# df = pd.DataFrame(data)
# df.to_csv('gameData.csv')
#
###########################################################################

df = pd.read_csv('gameData.csv')

# # funkcja mapująca
# def map_func(x, b, c, d, e, f, g):
#     return b * x["1"] + c * x["2"] + d * x["3"] + e * x["4"] + f * x["5"] + g
#
#
# #liczenie parametrów funkcji
# params, _ = curve_fit(map_func, xdata=df[["1", "2", "3", "4", "5"]], ydata=df["0"])
# mse = mean_squared_error(df["0"], map_func(df[["1", "2", "3", "4", "5"]], params[0], params[1],
#                                          params[2], params[3], params[4], params[5]))
#
# print(params)
# print(mse)

X_train, X_test, y_train, y_test = train_test_split(df["0"], df["5"])

LR = LinearRegression()
LR.fit(X_train.values.reshape(-1, 1), y_train)

prediction = LR.predict(X_test.values.reshape(-1, 1))

plt.title("Kill Participation")
plt.plot(X_test, prediction, label='Linear Regression', color='b')
plt.scatter(X_test, y_test, label='Actual Test Data', color='g', alpha=.7)
plt.legend()
plt.show()

print("Wygrana:")
print(LR.predict(np.array([[1]]))[0])
print("Przegrana:")
print(LR.predict(np.array([[0]]))[0])
# print(LR.score(X_test.values.reshape(-1, 1), y_test.values))
