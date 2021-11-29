from torchtext.legacy import data
from torchtext.legacy.datasets import SequenceTaggingDataset
from torchtext.vocab import FastText, CharNGram


#device="cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'


def conll2003_dataset(tag_type, batch_size, root='./conll2003',
                      train_file='eng.train',
                      validation_file='eng.testa',
                      test_file='eng.testb'):
    inputs_word = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True, lower=True)

    inputs_char_nesting = data.Field(tokenize=list, init_token="<bos>", eos_token="<eos>",
                                     batch_first=True)

    inputs_char = data.NestedField(inputs_char_nesting,
                                   init_token="<bos>", eos_token="<eos>")

    labels = data.Field(init_token="<bos>", eos_token="<eos>", batch_first=True)

    fields = ([(('inputs_word', 'inputs_char'), (inputs_word, inputs_char))] +
                [('labels', labels) if label == tag_type else (None, None)
                for label in ['pos', 'chunk', 'ner']])

    train, val, test = SequenceTaggingDataset.splits(
        path=root,
        train=train_file,
        validation=validation_file,
        test=test_file,
        separator=' ',
        fields=tuple(fields))

    inputs_char.build_vocab(train.inputs_char, val.inputs_char, test.inputs_char)
    inputs_word.build_vocab(train.inputs_word, val.inputs_word, test.inputs_word, max_size=50000,
                            vectors=[FastText(), CharNGram()])

    labels.build_vocab(train.labels)
    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=batch_size,
    device=device)
    train_iter.repeat = False

    return {
        'iterators': (train_iter, val_iter, test_iter),
        'vocabs': (inputs_word.vocab, inputs_char.vocab, labels.vocab)
    }

