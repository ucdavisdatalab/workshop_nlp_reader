Section Overview
================

This three-part workshop series introduces participants to natural language
processing (NLP) with Python. It builds on Getting Started with Textual Data by
extending the scope of data-inflected text analysis to include various methods
of modeling meaning. Sessions will cover NLP topics ranging from segmentation
and dependency parsing to sentiment analysis and context-sensitive modeling. We
will also discuss how to implement such methods for tasks like classification.
Basic familiarity with analyzing textual data in Python is required. We welcome
students, postdocs, faculty, and staff from a variety of research domains,
ranging from health informatics to the humanities.

Before We Begin...
==================

This reader is meant to serve both as a roadmap for the overall trajectory of
the series and as a reference for later work you may do in natural language
processing (NLP). Our sessions will follow its overall logic, but the reader
itself offers substantially more details than we may have time to discuss in
the sessions. The instructors will call attention to this when appropriate; you
are encouraged to consult the reader whenever you would like to learn more
about a particular topic.

Session schedule

| Session | Chapters Covered | Topic                                       |
| ------- | ---------------- | ------------------------------------------- |
|    1    |     Chapter 2    | Document annotation                         |
|    2    |     Chapter 3    | Text classification and feature engineering |
|    3    |     Chapter 4    | Word embeddings                             |


```{admonition} Learning Objectives
By the end of this series, you will be able to:

+ Use popular NLP frameworks in Python, including `Gensim` and `spaCy`
+ Explain key concepts and terminology in NLP, including dependency parsing,
  named entity recognition, and word embedding
+ Process texts to glean information about sentiment, subject, and style
+ Classify texts on the basis of their features
+ Produce models of word meanings from a corpus
+ Perform a few core NLP tasks including keyword analysis, relation extraction,
  document similarity analysis, and text summarization
```

File and Data Setup
-------------------

### Google Colab

We will be using Google Colab's platform and Google Drive during the series and
working with a set of pre-configured notebooks and data. You must have a Google
account to work in the Colab environment. Perform the following steps to setup
your environment for the course:

1. Download the [data][zipped]
2. Un-compress the downloaded .zip file by double clicking on it
3. Visit the Google Colab [website][site] at and sign-in using your Google
   account
4. In a separate browser tab (from the one where you are logged-in to Colab)
   sign-in to your Google Drive
5. Upload the `nlp_workshop_data` directory into the root of your Google Drive

[zipped]: https://datalab.ucdavis.edu/nlp_workshop_data/nlp_workshop_data.zip
[site]: https://colab.research.google.com

Once you have completed the above steps, you will have your basic environment
setup. Next, you'll need to create a blank notebook in Google Colab. To do
this, go to Google Colab and choose "File->New Notebook" from the File Menu.
Alternatively, select "New Notebook" in the bottom right corner of the
notebooks pop-up if it appears in your window.

Now, you need to connect your Google Drive to your Colab environment. To do
this, run the following code in the code cell at appears at the top of your
blank notebook:

```
from google.colab import drive
drive.mount('/gdrive')
```

Your environment should be ready to go!

### Template code

This workshop is hands-on, and you're encouraged to code alongside the
instructors. That said, we'll also start each session with some template code
from the session before. You can find these templates in this [start
script][ss] directory. Simply copy/paste the code from the `.txt` files into
your Jupyter environment.

[ss]: https://github.com/ucdavisdatalab/workshop_nlp_with_python/tree/main/start_scripts

Assessment
----------
If you are taking this workshop to complete a GradPathways [micro-credential
track][microcredential], you can find instructions for the assessment
[here][here].

[microcredential]:https://gradpathways.ucdavis.edu/micro-credentials
[here]: https://github.com/ucdavisdatalab/workshop_nlp_with_python/tree/main/assessment
