# TODO

## RAG

### Improve data cleaning for MD files

- [x] remove IXBRL header
- [x] remove tables
- [x] add csv tables

### Improve chunking

- [x] ensure tabular data is correctly embedded and retrieved
  
### Add summarization

- [x] add model as summarizer

### Add ability to load embeddings

- [x] add embedding loading (speeds up testing)
  
### Add filters

- [x] add filters for date and ticker

### Improve summarization

- [ ] add better model as summarizer (small thinking model finetuned in order to distill big thinking model)

## Orchestator

### Taskmaster

- [x] figure out parameters for each function
- [ ] cyclic execution of tasks till completion

### Agents

- [x] create agent class - takes in instructions, creates final parameters and executes
- [ ] implement the various agents that are needed

### Agents - specific

- [x] implement reporting agent short summarization
- [ ] implement reporting agent short report (more detailed summarization)
- [ ] implement reporting agent detailed report (presentations, reports, etc)

- [ ] implement retrieval agent yfinance search
- [ ] implement retrieval agent filing search
- [ ] implement retrieval agent fred search
- [ ] implement retrieval agent news search
- [ ] implement retrieval agent internet search

- [ ] implement trading agent strategizer
- [ ] implement trading agent strategy coder
- [ ] implement trading agent strategy backtester

- [ ] implement analysis agent strategy ratio calculator
- [ ] implement analysis agent strategy timeseries summary statistics

### Evaluator

- [ ] evaluate if task has been completed

### Pinpointer

Allows for models to retrieve session information easily - stores session data as a json file that is a serialized python dict, each object has a name, description and type (types allow for different types of summarization, eg semantic summarization with llm for textual data vs summary statistics for timeseries)

- [x] create tools for listing, retrieval, summarization
- [ ] create data summarization tooks (summary statistics)

## Tools

### Trading Engine

- [ ] create trading engine

### Search

- [ ] find a way to search efficiently

## LONGTERM

- [ ] create sequence to sequence model to turn queries into steps
- [ ] semantic caching of requests for analyzer and parameterizer
