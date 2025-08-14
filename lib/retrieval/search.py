import requests
from urllib.parse import urljoin

class SearchEngineWrapper:
    def __init__(self, base_url="https://northboot.xyz/", user_agent="Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"):
        self.base_url = base_url.rstrip("/")
        self.search_path = "/search"  # or "/"
        self.USER_AGENT = user_agent

    def search(self, query, *,
        categories=None,
        engines=None,
        language=None,
        page=1,
        time_range=None,
        safesearch=None,
        output_format="json",
        **extra_params):
        """
        Perform a synchronous search query.

        Args:
            query (str): Search query string (parameter 'q').
            categories (list[str], optional): e.g., ['general', 'images'].
            engines (list[str], optional): e.g., ['google', 'bing'].
            language (str, optional): e.g., "en", "all".
            page (int, optional): Page number, default 1.
            time_range (str, optional): 'day', 'month', 'year'.
            safesearch (int, optional): 0 = off, 1 = moderate, 2 = strict.
            output_format (str, optional): 'json', 'csv', 'rss'. Default 'json'.
            extra_params: Additional query parameters.

        Returns:
            dict or raw text: Parsed JSON if format is 'json', else raw response.
        """
        url = urljoin(self.base_url, self.search_path)
        params = {'q': query, 'format': output_format, 'pageno': page}

        if categories:
            params['categories'] = ",".join(categories)
        if engines:
            params['engines'] = ",".join(engines)
        if language:
            params['language'] = language
        if time_range:
            params['time_range'] = time_range
        if safesearch is not None:
            params['safesearch'] = safesearch

        params.update(extra_params)

        headers = {
            "User-Agent": self.USER_AGENT
        }

        response = requests.get(url, params=params, timeout=10, headers=headers)
        response.raise_for_status()

        # if output_format == "json":
        #     return response.json()
        # else:
        return response.text

se = SearchEngineWrapper()
print(se.search("What is the speed of light"))