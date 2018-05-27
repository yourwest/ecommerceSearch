import fasttext
import pandas as pd
from sklearn.utils import shuffle as shuffle_inplace

from common.utils import LoggerMixin
from vypryamitel.dataset import load_dataset_by_config
from .utils import check_and_touch_file
from .langutils import replace_punct

replace_punct = lambda s: replace_punct(s, more='_.')

class FasttextConstructing(LoggerMixin):
    def __init__(self, sites, path, model_name, switcher):
        for ext in ['', '.bin', '.vec']:
            check_and_touch_file(path + model_name + ext)

        def get_string_of_categories_names(prod_id):
            cats = main_categories_by_product[prod_id]
            return replace_punct(' '.join(
                filter(lambda s: not s.endswith('Нет в наличии'),
                       map(ds.get_category_name_normal, cats))))

        df = pd.DataFrame()
        for site in sites:
            self.debug('site: ' +  site)
            self.debug('dataset uploading')
            config = get_rr_site_config_by_name(site)
            ds = load_dataset_by_name(config, load_search_views=True, load_item_views=True)
            searches = ds.search_views[['externalsessionid', 'timestamp', 'searchstring']]
            views = ds.item_views[['externalsessionid', 'timestamp', 'product_id']]
            self.debug('getting name')
            main_categories_by_product = ds.get_main_categories_by_product()

            views['product_name'] = views['product_id'].apply(ds.get_product_name)
            views['category_name'] = views['product_id'].apply(get_string_of_categories_names)
            self.debug('switching')
            searches['searchstring'] = searches['searchstring'].apply(lambda s: switcher(str(s)))
            self.debug('punctuation')
            searches['searchstring'] = searches['searchstring'].apply(replace_punct)
            views['product_name'] = views['product_name'].apply(replace_punct)

            self.debug('grouping')
            search_text = searches[['externalsessionid', 'timestamp', 'searchstring']]
            search_text.columns = ['externalsessionid', 'timestamp', 'text']

            product_text = views[['externalsessionid', 'timestamp', 'product_name']]
            product_text = product_text[product_text['product_name'] != '<unknown>']
            product_text.columns = ['externalsessionid', 'timestamp', 'text']

            category_text = views[['externalsessionid', 'timestamp', 'category_name']]
            category_text.columns = ['externalsessionid', 'timestamp', 'text']

            df = pd.concat([search_text, product_text, category_text]).sort('timestamp')

            groups = df.groupby('externalsessionid')['text'].apply(lambda s: ' '.join(s.values))

            groups.to_csv(path + model_name, mode='a', index=None, header=None)

        df = pd.read_csv(path + model_name, delimiter='|', header=None)
        df[0] = df[0].str.replace('ё', 'е').str.replace(r'[ ]{2,}', ' ')
        shuffle_inplace(df)
        df.to_csv(path + model_name, sep='|', index=None, header=None)
        fasttext.skipgram(path + model_name, path + model_name)