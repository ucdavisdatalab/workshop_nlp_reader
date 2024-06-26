---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


Large Language Models: An Introduction
======================================

This chapter introduces large language models (LLMs). We will discuss
tokenization strategies, model architecture, the attention mechanism, and
dynamic embeddings. 

```{admonition} Learning objectives
By the end of this chapter, you should be able to:

+ Explain the difference between pretraining and fine-tuning
+ Have a general idea of how LLMs work
+ Know where to get LLMs for specific tasks
+ Explain what subwords are and tokenize texts with a subword tokenizer
+ Generate embeddings for texts from a LLM
+ Explain embedding pooling and retrieve different embedding representations
  from a model 
```


Preliminaries
-------------

We need the following libraries:

```{code-cell}
:tags: [remove-output]
import torch
from transformers import AutoTokenizer, AutoModel
```


### Using a pretrained model

```{margin} A note on terminology
In machine learning literature, you will see this idea of
pretraining/fine-tuning described as **transfer learning**.
```

Training LLMs requires vast amounts of data and computational resources. While
these resources are expensive, the very scale of these models contributes to
their ability to generalize. Practitioners will therefore use the same model
for a variety of tasks. They do this by **pretraining** a general model to
perform a foundational task, usually next-token prediction. Then, once that
model is trained, practitioners **fine-tune** that model for other tasks. The
fine-tuned variants benefit from the generalized language representations
learned during pretraining, but they adapt those representations to more
specific contexts and tasks.

The best place to find these pretrained models is [Hugging Face][hf]. The
company hosts thousands of them on its platform, and it also develops various
machine learning tools for working with these models. Hugging Face also
features fine-tuned models for various tasks, which may work out of the box for
your needs. Take a look at the [model listing][mlist] to see all models on the
platform. At the left, you'll see categories for model types, task types, and
more.

[hf]: https://huggingface.co/
[mlist]: https://huggingface.co/models

### Loading a model

To load a model from Hugging Face, first specify the **checkpoint** you'd like
to use. Typically this is just the name of the model.

```{code-cell}
checkpoint = "google-bert/bert-base-uncased"
```

The `transformers` library has different tokenizer and model classes for
different models/architectures and tasks. You can write these out directly, or
use the `Auto` classes, which dynamically determine what class you'll need for
a model and task. Below, we load the base BERT model without specifying a task.

```{code-cell}
:tags: [remove-output]
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
bert = AutoModel.from_pretrained(checkpoint)
```

If you don't have this model stored on your own computer, it will download
directly from Hugging Face. The default directory for storing Hugging Face data
is `~/.cache/hugggingface`. Set a `HF_HOME` environment variable from the
command line if you want Hugging Face downloads to default to a different
location on your computer.

```sh
export HF_HOME=/path/to/another/directory
```


Subword Tokenization
--------------------

Note that we have initialized a tokenizer and model from the same checkpoint.
This is important: LLMs depend on specific tokenizers, which are themselves
trained on corpus data before their corresponding models even see that data.
But why do tokenizers need to be trained in the first place?

The answer has to do with the highly general nature of LLMs. These models are
trained on huge corpora, which means they must represent millions of different
pieces of text. Model vocabularies would quickly balloon to a huge size if they
represented all these tokens, however, and at any rate this would both
inefficient and a waste of resources, since some tokens are extremely rare. In
traditional tokenization and model building, you'd set a cutoff below which
rare tokens could be ignored, but LLMs need _all_ text. That means they need to
represent every token in a corpus---without storing representations for every
token in a corpus.

Model developers square this circle by using pieces of words, or **subwords**,
to represent other tokens. That way, a model can literally spell out any text
sequence it needs to build without having representations for every unique
token in its training corpus. (This also means LLMs can handle text they've
never seen before.) Setting the cutoff for which tokens should be represented
in full and which are best represented by subwords requires training a
tokenizer to learn the token distribution in a corpus, build subwords, and
determine said cutoff.

With subword tokenization, the following phrase:

> large language models use subword tokenization

...becomes:

> large language models use sub ##word token ##ization

See the hashes? This tokenizer prepends them to its subwords. 


### Input IDs

The actual output of `transformers` tokenizer has a few parts. We use the
following sentence as an example:

```{code-cell}
sentence = "Then I tried to find some way of embracing my mother's ghost."
```

Send this to the tokenizer, setting the return type to PyTorch tensors. We also
return the attention mask.

```{code-cell}
inputs = tokenizer(
    sentence, return_tensors = "pt", return_attention_mask = True
)
```

**Input IDs** are the unique identifiers for every token in the input text.
These are what the model actually looks at.

```{code-cell}
inputs["input_ids"]
```

Use the `.decode()` method to transform an ID (or sequences of ids) back to
text.

```{code-cell}
tokenizer.decode(5745)
```

The tokenizer has entries for punctuation:

```{code-cell}
tokenizer.decode([1005, 1012])
```

Whitespace tokens are often removed, however:

```{code-cell}
ws = tokenizer(" \t\n")
ws["input_ids"]
```

But if that's the case, what are those two IDs? These are two special tokens
that BERT uses for a few different tasks.

```{code-cell}
tokenizer.decode([101, 102])
```

`[CLS]` is prepended to every input sequence. It marks the start of a sequence,
and it also serves as a "summarization" token for sequences, a kind of
aggregate representation of model outputs. When you train a model for
classification tasks, the model uses `[CLS]` to decide how to categorize a
sequence.

`[SEP]` is appended to every input sequence. It marks the end of a sequence,
and it is used to separate input pairs for tasks like sentence similarity,
question answering, and summarization. When training, a model looks to `[SEP]`
to distinguish which parts of the input correspond to task components.


### Token type IDs

Some models don't need anything more than `[CLS]` and `[SEP]` to make the above
distinctions. But other models also incorporate **token type IDs** to further
distinguish individual pieces of input. These IDs are binary values that tell
the model which parts of the input belong to what components in the task.

Our sentence makes no distinction:

```{code-cell}
inputs["token_type_ids"]
```

But a pair of sentences would:

```{code-cell}
question = "What did I do then?"
with_token_types = tokenizer(question, sentence)
with_token_types["token_type_ids"]
```


### Attention mask

A final output, **attention mask**, tells the model what part of the input
it should use when it processes the sequence.

```{code-cell}
inputs["attention_mask"]
```


### Padding and truncation

It may seem like a redundancy to add an attention mask, but tokenizers often
**pad** input sequences. While Transformer models can process sequences in
parallel, which massively speeds up their run time, each sequence in a
**batch** needs to be the same length. Texts, however, are rarely the same
length, hence the padding.

```{code-cell}
two_sequence_inputs = tokenizer(
    [question, sentence],
    return_tensors = "pt",
    return_attention_mask = True,
    padding = "longest"
)
two_sequence_inputs["input_ids"]
```

Token ID `0` is the `[PAD]` token.


```{code-cell}
tokenizer.decode(0)
```

Note the attention masks:

```{code-cell}
two_sequence_inputs["attention_mask"]
```

There are a few different strategies for padding. Above, we had the tokenizer
pad to the longest sequence in the input. But usually it's best to set it to
`max_length`:

```{code-cell}
:tags: [output_scroll]
two_sequence_inputs = tokenizer(
    [question, sentence],
    return_tensors = "pt",
    return_attention_mask = True,
    padding = "max_length"
)
two_sequence_inputs["input_ids"][0]
```

This will pad the text out to the maximum number of tokens the model can
process at once. This number is known as the **context window**.

```{code-cell}
print("Context window size:", tokenizer.model_max_length)
```

:::{warning}
Not all tokenizers have this information stored in their configuration. You
should always check whether this is the case before you use a tokenizer. If it
doesn't have this information, take a look at the model documentation.
:::

If your input exceeds the number above, you will need to **truncate** it,
otherwise the model may not process input properly.

```{code-cell}
too_long = "a " * 10_000
too_long_inputs = tokenizer(
    too_long, return_tensors = "pt", return_attention_mask = True
)
```

Set `truncation` to `True` to avoid this problem.

```{code-cell}
too_long_inputs = tokenizer(
    too_long,
    return_tensors = "pt",
    return_attention_mask = True,
    padding = "max_length",
    truncation = True
)
```

What if you have long texts, like novels? You'll need to make some decisions.
You could, for example, look for a model with a bigger context window; several
of the newest LLMs can process novel-length documents now. Or, you might
strategically chunk your text. Perhaps you're only interested in dialogue, or
maybe paragraph-length descriptions. You could preprocess your texts to create
chunks of this kind, ensure they do not exceed your context window size, and
then send them to the model.

Regardless of what strategy you use, it will take iterative tries to settle on
a final tokenization workflow.


Running the Model
-----------------

Now that we've tokenized our text, we can run our model. First, we move the
model to a device (like a GPU, represented by `0` below). The `transformers`
library is pretty good at doing this for us, but we can always set the device
explicitly

```{code-cell}
device = 0 if torch.cuda.is_available() else "cpu"
bert.to(device)

print("Moved model to", device)
```

:::{tip}
You can also set the model device when initializing it.

```py
model = AutoModel.from_pretrained(checkpoint, device = device)
```
:::

Time to process the inputs. First, put the model in evaluation mode. This
disables dropout, which can make outputs inconsistent (e.g. non-deterministic).

```{code-cell}
:tags: [output_scroll]
bert.eval()
```

Then, wrap the process in a context manager. This context manager will keep the
model from collecting gradients when it processes. Unless you are training a
model or trying understand model internals, there's no need for gradients. With
the context manager built, send the inputs to the model.

```{code-cell}
with torch.no_grad():
    outputs = bert(**inputs, output_hidden_states = True)
```


### Model outputs

There are several components in this output:

```{code-cell}
:tags: [output_scroll]
outputs
```

The `last_hidden_state` tensor contains the hidden states for each token after
the final layer of the model. Every vector is a contextualized representation
of a token. The shape of this tensor is (batch size, sequence length, hidden
state size).

```{code-cell}
:tags: [output_scroll]
outputs.last_hidden_state
```

The `pooler_output` tensor is usually the one you want to use if you are
embedding text to use for some other purpose. It corresponds to the hidden
state of the `[CLS]` token. Remember that models use this as a summary
representation of the entire sequence. The shape of this tensor is (batch size,
hidden state size).

```{code-cell}
:tags: [output_scroll]
outputs.pooler_output
```

Setting `output_hidden_states = True` had the model return all of the hidden
states, from the first embedding layer to the very last layer. These are
accessible from `hidden_states`. This is a tuple of tensors. Every tensor has
the shape (batch size, sequence length, hidden state size).

```{code-cell}
:tags: [output_scroll]
outputs.hidden_states
```

Other optional outputs, which we don't have here, include the following:

+ `past_key_values`: previously computed key and value matrices, which
  generative models can draw on to speed up computation
+ `attentions`: attention weights for every layer in the model
+ `cross_attentions`: layer-by-layer attention weights for models that work by
  attending to tokens across input pairs


### Which layer? Which token?

The next chapter will demonstrate a classification task with BERT. This
involves modifying the network layers to output one of a set of labels for
input. All this will happen inside the model itself, but it's also perfectly
fine to generate embeddings with a model and to use those embeddings for some
other task that has nothing to do with a LLM.

People often use the last hidden state embeddings for other tasks, though
there's no hard and fast rule saying that this is necessary. The
[BERTology paper][bertology] tells us that different layers in BERT do
different things: earlier ones capture syntactic features, while later ones
capture more semantic features. If you're studying syntax, you might choose an
earlier layer, or set of layers.

[bertology]: https://aclanthology.org/2020.tacl-1.54/

For general document embeddings, there are a number of options:

+ Instead of using the `[CLS]` token, **mean pooling** involves computing the
  mean of all tokens in the last hidden layer. This can potentially smooth out
  noise
+ **Max pooling** takes the max of all tokens' last hidden layer embeddings.
  This boosts salient features in a sequence
+ Other people compute the **mean of the last four layers** and select `[CLS]`
  from that (though you could use all tokens, too); others take the **sum of
  the last four layers**. Both strategies combine information from a greater
  portion of the model
+ A **concatenation of the last four layers** (like appending layers in a list)
  is yet another option. This can potentially combine different levels of
  abstraction

```{tip}
The [SentenceTransformers][st] package will handle many of these pooling
operations for you.

[st]: https://www.sbert.net/
```

Finally, while using `[CLS]` is customary, it's not necessary for all purposes
and you can select another token if you feel it would be better. You can even
train a classification model to learn from a different token, but be warned:
one of the reasons `[CLS]` is customary is because this token is in every input
sequence. The same cannot always be said of other tokens.

