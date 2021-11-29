import torch
import torch.nn as nn

START_TAG = "<bos>"
STOP_TAG = "<eos>"

device="cuda" if torch.cuda.is_available() else "cpu"


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_sum_exp_batch(log_Tensor, axis=-1):
    return torch.max(log_Tensor, axis)[0] + torch.log(
        torch.exp(log_Tensor - torch.max(log_Tensor, axis)[0].view(log_Tensor.shape[0], -1, 1)).sum(axis))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

    def init_hidden(self, sentence):
        return (torch.randn(2, sentence.shape[1], self.hidden_dim // 2),
                torch.randn(2, sentence.shape[1], self.hidden_dim // 2))

    def _forward_alg(self, feats):
        batch_size = feats.shape[0]
        log_alpha = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(device)
        log_alpha[:, 0, self.tag_to_ix[START_TAG]] = 0

        for t in range(1, feats.shape[1]):
            log_alpha = (log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        log_prob_all_barX = log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden(sentence)
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        batch_size = feats.shape[0]
        batch_transitions = self.transitions.expand(batch_size, self.tagset_size, self.tagset_size)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0], 1)).to(device)
        for t in range(1, feats.shape[1]):
            score = score + \
                    batch_transitions.gather(-1, (tags[:, t] * self.tagset_size + tags[:, t - 1]).view(-1, 1)) \
                    + feats[:, t].gather(-1, tags[:, t].view(-1, 1)).view(-1, 1)
        return score

    def _viterbi_decode(self, feats):
        batch_size = feats.shape[0]

        log_delta = torch.Tensor(batch_size, 1, self.tagset_size).fill_(-10000.).to(device)
        log_delta[:, 0, self.tag_to_ix[START_TAG]] = 0

        psi = torch.zeros((batch_size, feats.shape[1], self.tagset_size), dtype=torch.long).to(device)
        for t in range(1, feats.shape[1]):
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        path = torch.zeros((batch_size, feats.shape[1]), dtype=torch.long).to(device)

        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(feats.shape[1] - 2, -1, -1):
            path[:, t] = psi[:, t + 1].gather(-1, path[:, t + 1].view(-1, 1)).squeeze()

        return max_logLL_allz_allx, path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq
