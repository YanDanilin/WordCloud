from gensim.models import FastText
import pandas as pd
import gdown


class WordCloudFT(object):
    def __init__(self, path_to_model: str | None = None, path_to_litter: str = ''):
        '''
        Creating class object

        :param path_to_model: path to the model, if None author`s model will be downloaded and path will be 'modelWordCloud/synonyms_ft.model2'
        :param path_to_litter: path to csv file with words to be deleted
        '''
        if path_to_model is None:
            gdown.download_folder(
                'https://drive.google.com/drive/folders/1Q12TJnp_nYx5DqM-nqYiXGMBjhEpcrGF?usp=sharing')
            self.model = FastText.load('modelWordCloud/synonyms_ft.model2')
        else:
            self.model = FastText.load(path_to_model)
        self.markings = {'(прост.)', '(устар.)', '(офиц.)', '(книжн.)',
                         '(высок.)', '(разг.)', '(бран.)', '(шутл.)', '(обл.)', '(спец.)'}
        if len(path_to_litter) != 0:
            self.litter = set(pd.read_csv(path_to_litter)['0'].to_list())
        else:
            self.litter = {}

    def set_word_list(self, word_list: list | str):
        '''
        Setting list of words which word cloud will be made from

        :param word_list: list of words or path to csv file with words (the 1st row must be '0', other words written in column)
        '''
        if len(word_list) == 0:
            raise Exception("Word list is empty")
        if type(word_list) == str:
            if len(word_list) < 4:
                raise Exception("Wrong file name")
            if word_list[-4:] == '.csv':
                df = pd.read_csv(word_list)
                self.init_word_list = df['0'].to_list()
                self.word_list_len = len(self.init_word_list)
            else:
                raise Exception("Wrong file format")
        elif type(word_list) == list:
            self.init_word_list = word_list
            self.word_list_len = len(word_list)
        else:
            raise Exception("Wrong argument")

    def _word_cleaning(self, word: str) -> str:
        import re

        word = word.lower().strip()
        split_word = word.split()
        for lexem in split_word:
            if lexem in self.markings:
                split_word.remove(lexem)
        word1 = ' '.join(split_word)
        word_wo_punct = re.sub(r'[^\w\s-]', '', word1)
        split_word1 = word_wo_punct.split()
        for lexem in split_word1:
            if lexem in self.litter:
                split_word1.remove(lexem)
        res_word = '-'.join(split_word1)
        return res_word

    def _preprocess_word_list(self):
        self.word_list_preprocessed = []
        for index, w in enumerate(self.init_word_list):
            self.word_list_preprocessed.append(self._word_cleaning(w))

    def fit(self):
        import sklearn
        from sklearn.cluster import DBSCAN
        import numpy as np

        self._preprocess_word_list()
        self.vectors = np.array([self.model.wv[w]
                                for w in self.word_list_preprocessed])
        clustering = DBSCAN(eps=0.53, min_samples=2,
                            metric='cosine').fit(self.vectors)
        self.labels = clustering.labels_
        words_np = np.array(self.word_list_preprocessed)
        self.clusters = []
        self.cloud = {}
        for label in np.unique(self.labels):
            self.clusters.append(words_np[np.where(self.labels == label)[0]])
        for cluster in self.clusters:
            self.cloud[self.model.wv.rank_by_centrality(
                cluster)[0][1]] = len(cluster[0])

    def cloud_dict(self):
        '''
        Dictionary of words and count of words in cluster

        :return: dictionary of words and count of words in cluster
        '''
        return self.cloud

    def cloud_barchart(self):
        '''
        Displays horizontal barchart of cluster words and count of words in clusters
        '''
        import matplotlib.pyplot as plt

        y = list(self.cloud.keys())
        w = [self.cloud[i] for i in y]
        fig, ax = plt.subplots()
        bars = ax.barh(y=y, width=w)
        ax.bar_label(bars, padding=3)
        None

    def cloud_cloud(self, colormap='spring', background_color='#333333'):
        '''
        Displays cloud of words using wordcloud package
        '''
        from wordcloud import WordCloud as WC
        import matplotlib.pyplot as plt

        wordcloud = WC(background_color=background_color,
                       colormap=colormap,
                       min_font_size=10).generate_from_frequencies(self.cloud)
        wordcloud.to_image()
        plt.axis('off')
        plt.imshow(wordcloud)
        plt.tight_layout(pad=0)
        plt.figure(facecolor=None)
        plt.show()
