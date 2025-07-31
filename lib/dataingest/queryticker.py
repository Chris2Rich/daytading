import requests
from sec_edgar_api import EdgarClient
from markdownify import markdownify as md
from itertools import islice
import datetime
import random
import traceback
import argparse
import os
import sys
from concurrent import futures as cf
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set
from colorama import Fore, Style, init
from bs4 import BeautifulSoup
from arelle import Cntlr
import multiprocessing
import time
from threading import Lock
from collections import deque

sys.stdout.reconfigure(encoding='utf-8')
init(autoreset=True)

cntlr = Cntlr.Cntlr()
cntlr.webCache.workOffline = False

LIMIT = 3 #set to -1 if all

data_dir = Path("data/filings")
cik_map_path = Path("data/cik/cikmap.json")
error_log_path = Path("data/logs/queryticker_error.log")

USER_AGENT = "SECSCRAPER (default@default.com)"
client = EdgarClient(user_agent=USER_AGENT)
MAX_WORKERS = os.cpu_count()
FORMS = ("10-K", "10-Q", "8-K", "20-F", "40-F")

dl_executor = cf.ThreadPoolExecutor(max_workers=8)
successful_downloads = []
ibxrl_executor = cf.ProcessPoolExecutor(max_workers=MAX_WORKERS)
ibxrl_futures = []


class SlidingWindowRateLimiter:
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.lock = Lock()

    def acquire(self):
        with self.lock:
            now = time.monotonic()
            while self.calls and self.calls[0] <= now - self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                wait = self.period - (now - self.calls[0])
                time.sleep(wait)
                now = time.monotonic()
                while self.calls and self.calls[0] <= now - self.period:
                    self.calls.popleft()
            self.calls.append(now)

http_rate_limiter = SlidingWindowRateLimiter(max_calls=5, period=1.0)

def load_cik_map(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding='utf-8') as f:
        return json.load(f)

def format_cik(cik: str) -> str:
    return str(cik).zfill(10)

def strip_styles(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all(attrs={"style": True}):
        del tag["style"]
    return str(soup)

def convert_md_tables_to_csv_in_tags(markdown_text: str) -> str:
    lines = markdown_text.split('\n')
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        is_table_start = (
            line.strip().startswith('|') and
            i + 1 < len(lines) and
            '---' in lines[i+1] and
            lines[i+1].strip().startswith('|')
        )

        if is_table_start:
            table_lines = []
            table_lines.append(line)
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            
            csv_rows = []
            for tbl_line in table_lines:
                cleaned_line = tbl_line.strip()
                if cleaned_line.startswith('|'):
                    cleaned_line = cleaned_line[1:]
                if cleaned_line.endswith('|'):
                    cleaned_line = cleaned_line[:-1]
                
                cells = [cell.strip() for cell in cleaned_line.split('|')]
                csv_rows.append(','.join(cells))
            
            csv_content = '\n'.join(csv_rows)
            tabular_block = f"<TABULAR>\n{csv_content}\n</TABULAR>"
            new_lines.append(tabular_block)
        else:
            new_lines.append(line)
            i += 1
            
    return '\n'.join(new_lines)

def safe_fact_data(fact):
    context = fact.context
    concept = fact.concept

    end_date = getattr(context, "endDate", None)
    instant = getattr(context, "instantDatetime", None)

    # Determine periodType
    if instant:
        period_type = "instant"
    else:
        period_type = "forever"  # or unknown

    return {
        str(fact.qname): {
            "value": fact.value,
            "contextRef": fact.contextID,
            "unitRef": fact.unitID,
            "decimals": fact.decimals,
            "periodType": period_type,
            "endDate": str(end_date) if end_date else None,
            "instant": str(instant) if instant else None
        }
    }

def extract_ixbrl(html_path: Path, json_path: Path, error_log_filepath: Path) -> bool:
    try:
        modelXbrl = cntlr.modelManager.load(str(html_path))
        if not modelXbrl:
            error_detail_msg = "Arelle modelManager.load() returned None."
            raise RuntimeError(f"Arelle failed to load the document. Reason: {error_detail_msg}")

        records = []
        for fact in modelXbrl.facts:
            if "<table" in str(fact.value) or  "<div" in str(fact.value):
                continue
            records.append(safe_fact_data(fact))
        cntlr.modelManager.close(modelXbrl)
        records.sort(key=lambda x: list(list(x.values())[0].values())[1])
        json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"{Fore.YELLOW}[IXBRL DONE]{Style.RESET_ALL} {html_path} => {json_path}")
        return True
    except Exception as e:
        print(f"{Fore.RED}[IXBRL FAIL]{Style.RESET_ALL} {html_path}: {e}")
        with open(error_log_filepath, "a", encoding='utf-8') as f:
            f.write(f"IXBRL FAIL: {html_path} {e}\n")
        return False

def dl_processor(out_dir: Path, base_name: str, file_url: str, error_log_filepath: Path):
    html_path = out_dir / f"{base_name}.html"
    time.sleep(random.random() * .25)
    html_content = None
    start = time.time_ns()
    try:
        print(f"{Fore.BLUE}[HTTP GET]{Style.RESET_ALL} Waiting for slot for {file_url}")
        http_rate_limiter.acquire()
        print(f"{Fore.BLUE}[HTTP GET]{Style.RESET_ALL} Slot acquired. Fetching {file_url}")
        resp = requests.get(file_url, headers={"User-Agent": USER_AGENT})
        resp.raise_for_status()
        html_content = resp.text
        html_path.write_text(html_content, encoding='utf-8')
        print(f"{Fore.YELLOW}[HTML DONE]{Style.RESET_ALL} {file_url} => {html_path}")
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 429:
            print(f"{Fore.MAGENTA}[HTML RETRY]{Style.RESET_ALL} {file_url}")
            backoff = 5
            for _ in range(3):
                time.sleep(backoff + random.random())
                try:
                    return dl_processor(out_dir, base_name, file_url, error_log_filepath)
                except requests.HTTPError as e2:
                    backoff *= 2
                    continue
        raise
    except Exception as e:
        print(f"{Fore.RED}[HTML FAIL]{Style.RESET_ALL} {file_url}: {e}")
        with open(error_log_filepath, "a", encoding='utf-8') as f:
            f.write(f"HTML FAIL: {file_url} {e}\n")
        raise
    return
    
def ixbrl_processor(out_dir: Path, base_name: str, file_url: str, error_log_filepath: Path, parse_json : bool):
    if parse_json:
        (out_dir / "JSON").mkdir(parents=True, exist_ok=True)
    (out_dir / "MD").mkdir(parents=True, exist_ok=True)

    html_path = out_dir / f"{base_name}.html"
    json_path = out_dir / "JSON" / f"{base_name}.json"
    md_path = out_dir / "MD" / f"{base_name}.md"

    if parse_json:
        if not extract_ixbrl(html_path, json_path, error_log_filepath):
            return

    try:
        html_content = html_path.read_text(encoding='utf-8')
        soup = BeautifulSoup(html_content, 'lxml')
        ix_header = soup.find('ix:header')
        if ix_header:
            ix_header.decompose()
        cleaned_html = str(soup)
        markdown = convert_md_tables_to_csv_in_tags(md(cleaned_html))
        data_tag = f"<DATA location=\"{json_path.as_posix()}\" />"
        markdown = f"{data_tag}\n\n" + markdown
        md_path.write_text(markdown, encoding='utf-8')
        print(f"{Fore.YELLOW}[MD DONE]{Style.RESET_ALL} {file_url} => {md_path}")
    except Exception as e:
        print(f"{Fore.RED}[MD FAIL]{Style.RESET_ALL} {md_path}: {e}")
        with open(error_log_filepath, "a", encoding='utf-8') as f:
            f.write(f"MD FAIL: {md_path} {e}\n")
        raise
        return

    try:
        os.remove(html_path)
        print(f"{Fore.GREEN}[CLEANUP]{Style.RESET_ALL} Removed {html_path}")
    except OSError as e:
        print(f"{Fore.RED}[CLEANUP FAIL]{Style.RESET_ALL} Could not remove {html_path}: {e}")
        with open(error_log_filepath, "a", encoding='utf-8') as f:
            f.write(f"CLEANUP FAIL: {html_path} {e}\n")
        
    print(f"{Fore.GREEN}[DONE]{Style.RESET_ALL} Successfully processed {file_url}")

def download_filings(ticker: str, main_error_log_file_handle, global_error_log_path: Path, allowed_forms: Set[str] = set(FORMS)) -> None:
    ticker_up = ticker.upper()
    out_dir = data_dir / ticker_up
    out_dir.mkdir(parents=True, exist_ok=True)

    cik_map = load_cik_map(cik_map_path)
    if ticker_up not in cik_map:
        print(f"{Fore.RED}[CIK NF]{Style.RESET_ALL} CIK not found for ticker: {ticker_up}")
        main_error_log_file_handle.write(f"CIK NF: {ticker_up}\n")
        return
        
    cik = format_cik(cik_map[ticker_up]["cik"])

    print(f"{Fore.BLUE}[API CALL]{Style.RESET_ALL} Waiting for slot for SEC submissions API for CIK {cik}")
    print(f"{Fore.BLUE}[API CALL]{Style.RESET_ALL} Slot acquired. Fetching submissions for CIK {cik}")
    try:
        http_rate_limiter.acquire()
        submissions_data = client.get_submissions(cik=cik)
    except Exception as e:
        print(f"{Fore.RED}[API FAIL]{Style.RESET_ALL} Failed to get submissions for {ticker_up} (CIK {cik}): {e}")
        main_error_log_file_handle.write(f"API FAIL Submissions: {ticker_up} (CIK {cik}) {e}\n")
        return

    if "filings" not in submissions_data or "recent" not in submissions_data["filings"]:
        print(f"{Fore.RED}[API EMPTY]{Style.RESET_ALL} No recent filings found for {ticker_up} (CIK {cik})")
        main_error_log_file_handle.write(f"API EMPTY Submissions: {ticker_up} (CIK {cik})\n")
        return
        
    submissions = submissions_data["filings"]["recent"]
    filtered = {key: [] for key in submissions.keys()}

    for idx, date_str in enumerate(submissions.get("filingDate", [])):
        try:
            filing_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            main_error_log_file_handle.write(f"DATE PARSE FAIL: Invalid date '{date_str}' for {ticker_up} at index {idx}\n")
            continue

        form_type = submissions.get("form", [])[idx] if idx < len(submissions.get("form", [])) else None
        if form_type and filing_date.year >= 2010 and form_type in allowed_forms:
            for key, values in submissions.items():
                if idx < len(values):
                    filtered[key].append(values[idx])

    print(f"{Fore.YELLOW}[FILTER DONE]{Style.RESET_ALL} {ticker_up} => {cik}. Found {len(filtered.get('accessionNumber',[]))} relevant filings.")

    dl_futures = []
    infos = []
    for accession, doc_name, date_str, form_type in zip(
        filtered.get("accessionNumber",[]),
        filtered.get("primaryDocument",[]),
        filtered.get("filingDate",[]),
        filtered.get("form",[])
    ):
        safe_form = form_type.replace(" ", "_").replace("/", "-")
        accession_clean = accession.replace('-', '')
        base_name = f"{date_str}_{safe_form}_{accession_clean}"
        file_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{doc_name}"

        dl_futures.append(dl_executor.submit(dl_processor, out_dir, base_name, file_url, global_error_log_path))
        infos.append((out_dir, base_name, file_url))

    for future, (out_dir, base_name, file_url) in zip(dl_futures, infos):
        try:
            future.result()
            successful_downloads.append((out_dir, base_name, file_url))
        except Exception as exc:
            print(f"{Fore.RED}[TASK FAIL]{Style.RESET_ALL} {ticker_up} task exception: {exc}")
            main_error_log_file_handle.write(f"TASK EXCEPTION {ticker_up}: {exc}\n")
    print(f"{Fore.CYAN}[SUBMITTED ALL]{Style.RESET_ALL} All processing tasks for {ticker_up} submitted.")

def parse_filings(parse_json: bool = True):
    ibxrl_futures = [ibxrl_executor.submit(ixbrl_processor, out_dir, base_name, file_url, error_log_path, parse_json)for out_dir, base_name, file_url in successful_downloads]
    for future in cf.as_completed(ibxrl_futures):
        try:
            future.result()
        except Exception as exc:
            with open(error_log_path, "a", encoding='utf-8') as f:
                print(f"{Fore.RED}[TASK FAIL]{Style.RESET_ALL} {future} task exception: {exc}")
                f.write(f"TASK EXCEPTION {future}: {exc}\n")
    print(f"{Fore.GREEN}[COMPLETE]{Style.RESET_ALL} parsing has been completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process SEC filings.")
    parser.add_argument("ticker", help="Stock ticker, or 'SCRAPESEC' to process multiple tickers.")
    parser.add_argument("-dd", "--data_dir", help="Directory data will be put in", default="data/filings")
    parser.add_argument("-cd", "--cik_map_path", help="Path of cik mapping file", default="data/cik/cikmap.json")
    parser.add_argument("-ed", "--error_log_path", help="Path of error log", default="data/logs/queryticker_error.log")
    parser.add_argument("-l", "--limit", help="Number of tickers to scrape if using SCRAPESEC (-1 to scrape all)", default="3")
    parser.add_argument("-u", "--user_agent", help="user-agent for requests (organization (email))", default="SECSCRAPER (default@default.com)")
    parser.add_argument("-pj", "--parse_json", help="whether or not to parse json", default="True")
    args = parser.parse_args()

    LIMIT = int(args.limit) if args.limit else LIMIT
    data_dir = Path(args.data_dir) if args.data_dir else data_dir
    cik_map_path = Path(args.cik_map_path) if args.cik_map_path else cik_map_path
    error_log_path = Path(args.error_log_path) if args.error_log_path else error_log_path
    USER_AGENT = args.user_agent if args.user_agent else USER_AGENT
    parse_json = False if args.parse_json.lower() == "false" else True

    data_dir.parent.mkdir(parents=True, exist_ok=True)
    cik_map_path.parent.mkdir(parents=True, exist_ok=True)
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(error_log_path, "w", encoding='utf-8') as error_log_file_handle_main:
        try:
            if args.ticker.upper() == "SCRAPESEC":
                cik_map_data = load_cik_map(cik_map_path)
                tickers_to_process = list(islice(cik_map_data.keys(), LIMIT if LIMIT != -1 else len(cik_map_data.keys())))
                print(f"{Fore.GREEN}SCRAPESEC mode: processing up to {LIMIT} tickers.{Style.RESET_ALL}")
                for t in tickers_to_process:
                    raw_data_path_for_ticker = data_dir / t.upper()
                    print(f"{Fore.CYAN}[PROCESSING]{Style.RESET_ALL} Starting download for ticker: {t}")
                    download_filings(t, error_log_file_handle_main, error_log_path)
                    time.sleep(2)
            else:
                print(f"{Fore.CYAN}[PROCESSING]{Style.RESET_ALL} Starting download for ticker: {args.ticker}")
                download_filings(args.ticker, error_log_file_handle_main, error_log_path)
            
            print(f"{Fore.GREEN}All tickers have been downloaded.{Style.RESET_ALL}")

        except Exception as e:
            print(f"{Fore.RED}[CRITICAL ERROR]{Style.RESET_ALL} An error occurred in the main execution block: {e}")
            error_log_file_handle_main.write(f"CRITICAL MAIN ERROR: {e}\n")
            traceback.print_exc(file=error_log_file_handle_main)
        finally:
            dl_executor.shutdown(wait=True)
            parse_filings(parse_json)
            ibxrl_executor.shutdown(wait=True)
            print(f"{Fore.GREEN}[COMPLETE]{Style.RESET_ALL} All operations finished. Executor shut down.")
    
    print(f"Execution complete. Check {error_log_path} for any errors.")
    time.sleep(5)