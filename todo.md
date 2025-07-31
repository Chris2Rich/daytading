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

- [ ] break down query
- [ ] figure out what is needed (retrieval/report/strategize)
- [ ] cyclic execution of tasks till completion

### Evaluator

- [ ] evaluate if task has been completed

### Pinpointer
Allows for models to retrieve session information easily - stores session data as a json file that is a serialized python dict, each object has a name, description and type (types allow for different types of summarization, eg semantic summarization with llm for textual data vs summary statistics for timeseries)

- [ ] create tools for listing, retrieval, summarization