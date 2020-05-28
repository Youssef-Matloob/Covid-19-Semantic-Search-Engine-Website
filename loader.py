import numpy as np 
import pandas as pd 
import glob
import json
import gc
import matplotlib.pyplot as plt

plt.style.use('ggplot')
pd.set_option('display.max_columns', 30)

root_path = '/home/mohamed/Desktop/Codes/Covid-Data'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})

#print(meta_df.head())

all_json = glob.glob(f'{root_path}/**/**/*.json', recursive=True)

#print(len(all_json))

class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            #print(content.keys())
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            try :
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            except Exception as e:
                pass
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)

                
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
#first_row = FileReader(all_json[0])

#print(first_row)

class loader():
    def __init__(self):
        pass

    @staticmethod
    def get_breaks(content, length):
        data = ""
        words = content.split(' ')
        total_chars = 0

        # add break every length characters
        for i in range(len(words)):
            total_chars += len(words[i])
            if total_chars > length:
                data = data + "<br>" + words[i]
                total_chars = 0
            else:
                data = data + " " + words[i]
        return data

    def load(self,start = 0, end = 1000):
        dict_ = {'paper_id': [], 'abstract': [], 'body_text': [],'publish_year':[],'url':[],'WHO #Covidence':[] ,'license':[],'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
        for idx, entry in enumerate(all_json[:end]):
            #print(dict_)
            if idx % (len(all_json) // 100) == 0:
                print(f'Processing index: {idx} of {len(all_json)}')
            content = FileReader(entry)
            # get metadata information
            #print(content.paper_id)
            meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
            #print("meta data ", meta_data)
            # no metadata, skip this paper
            if len(meta_data) == 0:
                #print("%%%%%%%%%%%%%%%%%55")
                continue
            
            dict_['paper_id'].append(content.paper_id)
            dict_['abstract'].append(content.abstract)
            dict_['body_text'].append(content.body_text)
            dict_['publish_year'].append(pd.DatetimeIndex(meta_data['publish_time']).year.values[0])   
            dict_['url'].append(meta_data['url'].values[0])
            dict_['WHO #Covidence'].append(meta_data['WHO #Covidence'].values[0])
            dict_['license'].append(meta_data['license'].values[0])
            # also create a column for the summary of abstract to be used in a plot
            if len(content.abstract) == 0: 
                # no abstract provided
                dict_['abstract_summary'].append("Not provided.")
            elif len(content.abstract.split(' ')) > 100:
                # abstract provided is too long for plot, take first 300 words append with ...
                info = content.abstract.split(' ')[:100]
                summary = self.get_breaks(' '.join(info), 40)
                dict_['abstract_summary'].append(summary + "...")
            else:
                # abstract is short enough
                summary = self.get_breaks(content.abstract, 40)
                dict_['abstract_summary'].append(summary)
                
            # get metadata information
            meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
            
            try:
                # if more than one author
                authors = meta_data['authors'].values[0].split(';')
                if len(authors) > 2:
                    # more than 2 authors, may be problem when plotting, so take first 2 append with ...
                    dict_['authors'].append(". ".join(authors[:2]) + "...")
                else:
                    # authors will fit in plot
                    dict_['authors'].append(". ".join(authors))
            except Exception as e:
                # if only one author - or Null valie
                dict_['authors'].append(meta_data['authors'].values[0])
            
            # add the title information, add breaks when needed
            try:
                title = self.get_breaks(meta_data['title'].values[0], 40)
                dict_['title'].append(title)
            # if title was not provided
            except Exception as e:
                dict_['title'].append(meta_data['title'].values[0])
            
            # add the journal information
            dict_['journal'].append(meta_data['journal'].values[0])
        #print(dict_)    
        
        df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'publish_year','url','license','WHO #Covidence','body_text', 'authors', 'title', 'journal', 'abstract_summary'])

        return df_covid

#df_covid.head()


#print(df_covid.body_text)