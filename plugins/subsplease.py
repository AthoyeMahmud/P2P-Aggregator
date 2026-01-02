#VERSION: 1.0
# AUTHORS: PlutoMonkey

from html.parser import HTMLParser
from helpers import download_file, retrieve_url
from novaprinter import prettyPrinter
# some other imports if necessary
import json
from urllib.parse import urlparse
from urllib.parse import parse_qs

class subsplease(object):
    url = 'https://subsplease.org/'
    name = 'SubsPlease'
    supported_categories = {'all': ''}

    def search(self, what, cat='all'):
        for page in range(6):
            search_url = f"https://subsplease.org/api/?f=search&tz=$&s={what}&p={page}"
            response = retrieve_url(search_url)
            response_json = json.loads(response)
            if isinstance(response_json, dict):
                items = response_json.items()
            elif isinstance(response_json, list):
                items = []
                for entry in response_json:
                    if isinstance(entry, dict):
                        name = entry.get("show") or entry.get("name") or entry.get("title") or "SubsPlease"
                        items.append((name, entry))
            else:
                continue

            for result_name, result_data in items:
                downloads = result_data.get("downloads") if isinstance(result_data, dict) else None
                if not isinstance(downloads, list):
                    continue
                for download in downloads:
                    magnet_link = download.get("magnet")
                    if not magnet_link:
                        continue
                    parsed_url = urlparse(magnet_link)
                    size_list = parse_qs(parsed_url.query).get('xl')
                    size = size_list[0] if size_list else ""

                    res = {'link': magnet_link,
                        'name': f"[SubsPlease] {result_name} ({download['res']}p)",
                        'size': size,
                        'seeds': '-1',
                        'leech': '-1',
                        'engine_url': search_url,
                        'desc_link': '-1'}
                    prettyPrinter(res)
        
