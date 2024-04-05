# Workshop: Text Analysis and Natural Language Processing for Data Science

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC_BY--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

_[UC Davis DataLab](https://datalab.ucdavis.edu/)_  
Spring 2024
_Instructor: Tyler Shoemaker <<tshoemaker@ucdavis.edu>>, Carl Stahmer <<cstahmer@ucdavis.edu>>_
_Maintainer: Tyler Shoemaker <<tshoemaker@ucdavis.edu>>_  

* [Reader](https://ucdavisdatalab.github.io/workshop_nlp_reader/)
* [Event Page](https://datalab.ucdavis.edu/eventscalendar/YOUR_EVENT/)

This week-long workshop series covers the basics of text mining and natural
language processing (NLP) with Python. We will focus primarily on unstructured
text data, discussing how to format and clean text to enable the discovering of
significant patterns in collections of documents. Sessions will introduce
participants to core terminology in text mining/NLP and will walk through
methods that range from tokenization and dependency parsing to text
classification, topic modeling, and word embeddings. Basic familiarity with
Python is required. We welcome students, postdocs, faculty, and staff from a
variety of research domains, ranging from health informatics to the humanities.

Note: this series concludes with a special session on large language models,
"The Basics of Large Language Models."

By the end of this series, you will be able to:
+ Clean and structure textual data for analysis Recognize and explain how these
  cleaning processes impact research findings
+ Explain key concepts and terminology in text mining/NLP, including
  tokenization, dependency parsing, word embedding
+ Use special data structures such as document-term matrices to efficiently
  analyze multiple texts
+ Use statistical measures (pointwise mutual information, tf-idf) to identify
  significant patterns in text
+ Classify texts on the basis of their features
+ Produce statistical models of topics from/about a collection of texts
+ Produce models of word meanings from a corpus


## Contributing

The course reader is a live webpage, hosted through GitHub, where you can enter
curriculum content and post it to a public-facing site for learners.

To make alterations to the reader:
	  
1.  Check in with the reader's current maintainer and notify them about your 
    intended changes. Maintainers might ask you to open an issue, use pull 
    requests, tag your commits with versions, etc.

2.  Run `git pull`, or if it's your first time contributing, see
    [Setup](#setup).

3.  Edit an existing chapter file or create a new one. Chapter files may be 
    either Markdown files (`.md`) or Jupyter Notebook files (`.ipynb`). Either 
    is fine, but you must remain consistent across the reader (i.e. don't mix 
    and match filetypes). Put all chapter filess in the `chapters/` directory.
    Enter your text, code, and other information directly into the file. Make 
    sure your file:

    - Follows the naming scheme `##_topic-of-chapter.md/ipynb` (the only 
      exception is `index.md/ipynb`, which contains the reader's front page).
    - Begins with a first-level header (like `# This`). This will be the title
      of your chapter. Subsequent section headers should be second-level
      headers (like `## This`) or below.

    Put any supporting resources in `data/` or `img/`.

4.  Run the command `jupyter-book build .` in a shell at the top level of the
    repo to regenerate the HTML files in the `_build/`.

5.  When you're finished, `git add`:
    - Any files you edited directly
    - Any supporting media you added to `img/`

    Then `git commit` and `git push`. This updates the `main` branch of the
    repo, which contains source materials for the web page (but not the web
    page itself).

6.  Run the following command in a shell at the top level of the repo to update
    the `gh-pages` branch:
    ```
    ghp-import -n -p -f _build/html
    ```
    This uses the [`ghp-import` Python package][ghp-import], which you will
    need to install first (`pip install ghp-import`). The live web page will
    update automatically after 1-10 minutes.

[ghp-import]: https://github.com/c-w/ghp-import


## Setup

### Python Packages

We recommend using [conda][] to manage Python dependencies. The `env.yaml` file
in this repo contains a list of packages necessary to build the reader. You can
create a new conda environment with all of the packages listed in that file
with this shell command:

```sh
conda env create --file env.yaml
```

[conda]: https://docs.conda.io/en/latest/
