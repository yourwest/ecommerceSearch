import codecs
import os
import shutil
from common.utils import *
from common.feed_downloader import FeedDownloader
from vypryamitel.config import get_search_site_config_by_name, get_site_config_by_name


def unzip_file(zip, zip_file_info, filename):
    with codecs.open(filename, "wb") as f_out:
        with zip.open(zip_file_info) as f_in:
            shutil.copyfileobj(f_in, f_out)


def get_table_dump_date(filename):
    # Gross ugly hacks ahead!
    # assumes that file name is ALWAYS <timestamp-XXX-XXX-XXX>
    return timestamp2date(filename.split("-")[0])


@exception_handler("... Feeds updating failed")
def update_site_feeds(site):
    today = (dt.date.today() - dt.timedelta(days=2)).strftime("%Y_%m_%d")
    logging.getLogger('daps').info("Updating feeds for " + site.name)
    if not os.path.isdir(site.dataset_path):
        os.mkdir(site.dataset_path)
    FeedDownloader(site, today).run(False, site.dataset_path)
    logging.getLogger('daps').info("... done")


def update_feed_by_name(site):
    search_config = get_search_site_config_by_name(site)
    config = get_site_config_by_name(site)
    if search_config.type == 'rr' and config is not None:
        update_site_feeds(config)
    return
