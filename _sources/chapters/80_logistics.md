Section Overview
================ 

This workshop focuses on the basics of working with large language models
(LLMs) as part of the research pipeline, including understanding and
interrogating the models themselves as well as interacting with their
generative capabilities. Specific topics will include: setting up your own
open-source LLM, fine-tuning LLMs, and the basics of prompt engineering.


File and Data Setup
-------------------

We will be using Google Colab's platform and Google Drive during the workshop.
You must have a Google account to work in the Colab environment. Perform the
following steps to setup your environment for the course:

1. Download the [data][zipped]
2. Un-compress the downloaded .zip file by double clicking on it
3. Visit the Google Colab [website][site] at and sign-in using your Google
   account
4. In a separate browser tab (from the one where you are logged-in to Colab)
   sign-in to your Google Drive
5. Upload the `nlp_workshop_data` directory into the root of your Google Drive

[zipped]: https://datalab.ucdavis.edu/nlp_workshop_data/
[site]: https://colab.research.google.com

Once you have completed the above steps, you will have your basic environment
setup. Next, you'll need to create a blank notebook in Google Colab. To do
this, go to Google Colab and choose "File->New Notebook" from the File Menu.
Alternatively, select "New Notebook" in the bottom right corner of the
notebooks pop-up if it appears in your window.

Now, you need to connect your Google Drive to your Colab environment. To do
this, run the following code in the code cell at appears at the top of your
blank notebook:

```py
from google.colab import drive
drive.mount('/content/drive')
```

Your environment should be ready to go!


