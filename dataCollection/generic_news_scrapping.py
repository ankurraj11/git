import requests
import pandas as pd
import json
import re
import certifi

import urllib3
from bs4 import BeautifulSoup as bs

from newsapi import NewsApiClient
import datetime 
from newspaper import Article

ankur92 = '898565eccada4edda31b8781680ffe79'
raj786 = '6de62ac82944426da0c9a47ac0fd6a1d'
pra = '5473016e1f0d45919c6cf50d9568f598'

newsapi = NewsApiClient(api_key=ankur92)

state = 'tamilnadu'
path_to_text = '/Users/AR/Desktop/something/data/textFiles/'+state+'/'
path_to_csv = '/Users/AR/Desktop/something/data/urls_csv/'

def newsData(query, start_year, start_month, start_day, end_year, end_month, end_day):

	query_string = query.replace(' ','')
	csv_path = path_to_csv+query_string+'.csv'
	csv = pd.read_csv(csv_path, sep = ',', error_bad_lines=False, encoding = 'ISO-8859-1')

	start_time = datetime.datetime(start_year,start_month,start_day)
	final_end_time = datetime.datetime(end_year,end_month,end_day)

	end_time = start_time + datetime.timedelta(days=1)

	for j in range((final_end_time - start_time).days):

		print('start_day:',start_time, '--- end_day:', end_time)
		
		query_output = newsapi.get_everything(q= query,language='en',sort_by='relevancy',from_parameter= str(start_time)[:10],to= str(end_time)[:10])

		# try:
		for i in range(0,len(query_output['articles'])):

			print(len(query_output['articles']))
			query_output_list = []

			if query_output['articles'][i]['url'] in list(csv['url']):
				continue

			else:
				query_output_list.append([query_output['articles'][i]['publishedAt'], query_output['articles'][i]['title'], query_output['articles'][i]['url'], query_output['articles'][i]['description']])

				query_output_df = pd.DataFrame(query_output_list, columns = ['publishedAt','title','url','description'])

				with open(csv_path, 'a') as f:
					query_output_df.to_csv(f, header=False, index = False)

		# except:
		# 	start_time = start_time + datetime.timedelta(days=2)
		# 	end_time = end_time + datetime.timedelta(days=2)
		# 	continue

		start_time = start_time + datetime.timedelta(days=2)
		end_time = end_time + datetime.timedelta(days=2)

		if(start_time > final_end_time):
			break
		        		

	return query_string



def scrap(query_string):

	# dataframe = frame
	dataframe = pd.read_csv(path_to_csv+query_string+'.csv', encoding = 'ISO-8859-1')

	for i in range(dataframe.shape[0]):
		try:
			url = dataframe.iloc[i]['url']
			date = dataframe.iloc[i]['publishedAt'][:10]

			article = Article(url)   
			article.download()
			article.parse()

			text_content = article.text.replace('\n', '')

			if not text_content:
				continue

			else:
				text_file = open(path_to_text+date+'_'+query_string+'_'+str(i)+'.txt','w') 
				text_file.write(text_content)
				text_file.close()

			print(i)
		except:
			continue

query_string = newsData('punjab politics', 2018, 1, 1, 2018, 7, 31)
#scrap(query_string,frame)
## sources are { }





